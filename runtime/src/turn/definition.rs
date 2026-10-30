//! The chat turn as one durable Holdpoint definition (`agent_runtime.AgentRuntime` on LangGraph's `create_agent`).
//!
//! The loop is the Python agent's: call the model; refuse a cut-short answer; review the answer with the guards in
//! the order LangGraph ran their `after_model` hooks (the reverse of their registration); answer a refused answer's
//! every call with the guard's correction and call again; otherwise run its calls (local tools answer at once, Action
//! and Browser calls become one durable batch of Team requests the run waits on); end on a plain reply, a recorded
//! clarification, or a recorded Routine; and finish a completed turn with the independent memory check. Every model
//! call and Team request is a journaled step, so a resumed or recovered run replays recorded answers and never asks
//! Team twice. The definition is deterministic outside its `Ctx`: ids derive from the run id and positions.

use holdpoint::{Ctx, Definition, Interrupt, Outcome};
use sha2::{Digest, Sha256};

use crate::budget;
use crate::conversation::{AssistantMessage, Message, ToolMessage};
use crate::json::{self, Value};
use crate::tools::{self, clarification, memory, routine};
use crate::turn::run::{self, TeamRequest, TurnInput};
use crate::turn::step;

/// The graph steps one invocation (a start or a resume) may take, as LangGraph's `recursion_limit` bounded them.
pub const RECURSION_LIMIT: u32 = 12;

/// The chat turn definition.
#[derive(Debug, Default)]
pub struct BrainTurn;

/// Why a turn ended without a result, as the HTTP boundary answers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failed {
    /// A contract error with its fixed text (400).
    Contract(String),
    /// A provider failure (502).
    Provider,
    /// The turn was cancelled (409).
    Cancelled,
}

impl Failed {
    fn value(&self) -> Value {
        let (kind, detail) = match self {
            Self::Contract(detail) => ("contract", detail.as_str()),
            Self::Provider => ("provider", ""),
            Self::Cancelled => ("cancelled", ""),
        };
        json::object([(
            "error",
            json::object([("kind", json::text(kind)), ("detail", json::text(detail))]),
        )])
    }

    /// The failure a turn outcome names, if it is one.
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        let error = value.get("error")?;
        match json::str_of(error, "kind")? {
            "contract" => Some(Self::Contract(json::str_of(error, "detail")?.to_owned())),
            "provider" => Some(Self::Provider),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

impl From<step::Stop> for Result<Outcome, Interrupt> {
    fn from(stop: step::Stop) -> Self {
        match stop {
            step::Stop::Interrupt(interrupt) => Err(interrupt),
            step::Stop::Failed(failed) => Ok(Outcome::Done {
                result: run::outcome_json(&failed.value()),
            }),
        }
    }
}

impl Definition for BrainTurn {
    fn id(&self) -> &str {
        run::DEFINITION
    }

    fn version(&self) -> u32 {
        1
    }

    fn run(&self, ctx: &mut Ctx<'_>) -> Result<Outcome, Interrupt> {
        let Some(input) = TurnInput::from_json(ctx.input()) else {
            return Err(Interrupt::Unrecoverable("invalid-input".into()));
        };
        let mut turn = Turn::new(ctx.run_id(), input);
        match turn.drive(ctx) {
            Ok(result) => Ok(Outcome::Done {
                result: run::outcome_json(&result),
            }),
            Err(stop) => stop.into(),
        }
    }
}

/// One run's state as the definition re-executes it.
pub struct Turn {
    /// The run's input.
    pub input: TurnInput,
    /// The conversation so far.
    pub messages: Vec<Message>,
    prefix: String,
    counter: u32,
    /// Graph steps taken in the current invocation.
    pub steps: u32,
    /// Whether a clarification (and a memory change) may still be asked: only before the first Team result.
    pub first_segment: bool,
    /// Whether the last Team result answered the Browser: the model may then only reply.
    pub after_browser: bool,
}

impl Turn {
    fn new(run_id: &str, input: TurnInput) -> Self {
        let digest = Sha256::digest(run_id.as_bytes());
        let prefix = crate::text::hex(&digest[..8]);
        Self {
            messages: input.messages.clone(),
            input,
            prefix,
            counter: 0,
            steps: 0,
            first_segment: true,
            after_browser: false,
        }
    }

    /// The next deterministic id within the run.
    pub fn next_id(&mut self, kind: &str) -> String {
        self.counter += 1;
        format!("brain-{}-{kind}{}", self.prefix, self.counter)
    }

    /// Counts graph steps of the current invocation, ending the turn past the recursion limit.
    ///
    /// # Errors
    ///
    /// [`step::Stop::Failed`] with a provider failure, as LangGraph's recursion error became.
    pub fn take_steps(&mut self, steps: u32) -> Result<(), step::Stop> {
        self.steps += steps;
        if self.steps > RECURSION_LIMIT {
            Err(step::Stop::Failed(Failed::Provider))
        } else {
            Ok(())
        }
    }

    fn drive(&mut self, ctx: &mut Ctx<'_>) -> Result<Value, step::Stop> {
        loop {
            let message = step::call_model(self, ctx)?;
            self.messages.push(Message::Assistant(message));
            match self.review()? {
                Reviewed::Refused(correction) => {
                    self.answer_all(&correction);
                    continue;
                }
                Reviewed::Recorded(content) => {
                    self.answer_lone(routine::TOOL_NAME, content);
                    return self.finish(ctx);
                }
                Reviewed::Run => {}
            }
            let latest = self.latest().cloned().unwrap_or_else(empty_turn);
            if latest.calls.is_empty() {
                return self.finish(ctx);
            }
            self.take_steps(1)?;
            if step::run_tools(self, ctx, &latest)? {
                return self.finish(ctx);
            }
        }
    }

    /// The latest model turn.
    #[must_use]
    pub fn latest(&self) -> Option<&AssistantMessage> {
        self.messages.last().and_then(Message::assistant)
    }

    fn review(&mut self) -> Result<Reviewed, step::Stop> {
        let context = &self.input.context;
        let mut guards: Vec<Guard> = Vec::new();
        if self.after_browser {
            guards.push(Guard::AfterBrowser);
        }
        if self.input.browser {
            guards.push(Guard::BrowserAlone);
        }
        if context.routine_rerun.is_some() {
            guards.push(Guard::Rerun);
        }
        if self.input.routine_tool {
            guards.push(Guard::Routine);
        }
        if self.input.memory_tool {
            guards.push(Guard::Memory);
        }
        guards.push(Guard::Clarification);
        for guard in guards {
            self.take_steps(1)?;
            match self
                .apply(guard)
                .map_err(|error| step::Stop::Failed(step::failed_of(&error)))?
            {
                Reviewed::Run => {}
                other => return Ok(other),
            }
        }
        Ok(Reviewed::Run)
    }

    fn apply(&self, guard: Guard) -> Result<Reviewed, crate::errors::BrainError> {
        let messages = &self.messages;
        let context = &self.input.context;
        let refused = |review: tools::Review| {
            review.map(|verdict| verdict.map_or(Reviewed::Run, Reviewed::Refused))
        };
        match guard {
            Guard::AfterBrowser => refused(tools::browser::review_after(messages)),
            Guard::BrowserAlone => refused(tools::browser::review_alone(messages)),
            Guard::Rerun => {
                let work = context.routine_rerun.clone().unwrap_or_default();
                refused(routine::rerun_review(
                    messages,
                    &work,
                    &step::action_names(&self.input),
                ))
            }
            Guard::Routine => match routine::review(messages, context)? {
                routine::Verdict::Pass => Ok(Reviewed::Run),
                routine::Verdict::Refuse(correction) => Ok(Reviewed::Refused(correction)),
                routine::Verdict::Record(record) => Ok(Reviewed::Recorded(record)),
            },
            Guard::Memory => refused(memory::review(messages, self.first_segment)),
            Guard::Clarification => refused(clarification::review(
                messages,
                self.first_segment,
                context.routine_mode,
                context.locale.as_deref(),
            )),
        }
    }

    /// Answers every call of the latest model turn with `content` (`tool_refusal.refusal`).
    fn answer_all(&mut self, content: &str) {
        let calls = self
            .latest()
            .map(|message| message.calls.clone())
            .unwrap_or_default();
        for call in calls {
            let id = self.next_id("t");
            self.messages.push(Message::Tool(ToolMessage {
                id,
                call_id: call.id,
                name: call.name,
                content: content.to_owned(),
                error: false,
            }));
        }
    }

    fn answer_lone(&mut self, name: &str, content: String) {
        let call_id = self
            .latest()
            .and_then(|message| message.calls.first())
            .map(|call| call.id.clone())
            .unwrap_or_default();
        let id = self.next_id("t");
        self.messages.push(Message::Tool(ToolMessage {
            id,
            call_id,
            name: name.to_owned(),
            content,
            error: false,
        }));
    }

    /// Records the turn's answer to a set of Team requests, in call order, and starts a new invocation.
    pub fn resumed(&mut self, browser: bool) {
        self.steps = 0;
        self.first_segment = false;
        self.after_browser = browser;
    }

    fn finish(&mut self, ctx: &mut Ctx<'_>) -> Result<Value, step::Stop> {
        let result = step::settle(self)?;
        let memory = step::check_memory(self, ctx, &result)?;
        Ok(json::object([
            ("turn", step::turn_value(&result, &memory)),
            ("messages", crate::conversation::to_value(&self.messages)),
        ]))
    }

    /// The next request's interrupt id: 32 hex digits derived from the run and the request's position, so a replay
    /// asks Team under the same ids (`turn.SAFE_ID_RE`).
    pub fn interrupt_id_next(&mut self) -> String {
        self.counter += 1;
        let digest = Sha256::digest(format!("{}:i{}", self.prefix, self.counter).as_bytes());
        crate::text::hex(&digest[..16])
    }

    /// The fixed tokens of this turn's prompt and tools (`_fixed_tokens`).
    #[must_use]
    pub fn fixed_tokens(&self) -> u64 {
        let tools: Vec<Value> = self.input.tools.iter().map(run::tool_value).collect();
        budget::fixed_tokens(&self.input.system, &tools) + self.input.context.attachments.charge
    }

    /// Builds a [`TeamRequest`] list into the run's deferred step.
    #[must_use]
    pub fn requests_json(requests: &[TeamRequest]) -> Vec<(&'static str, serde_json::Value)> {
        requests
            .iter()
            .map(|request| (request.name(), request.to_json()))
            .collect()
    }
}

fn empty_turn() -> AssistantMessage {
    AssistantMessage {
        id: String::new(),
        text: String::new(),
        calls: vec![],
        invalid_calls: vec![],
        refusal: false,
        truncated: false,
        provider: String::new(),
        raw: None,
        attempts: 1,
    }
}

#[derive(Debug, Clone, Copy)]
enum Guard {
    AfterBrowser,
    BrowserAlone,
    Rerun,
    Routine,
    Memory,
    Clarification,
}

enum Reviewed {
    Run,
    Refused(String),
    Recorded(String),
}
