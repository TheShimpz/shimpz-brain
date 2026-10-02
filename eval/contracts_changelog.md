# Arm-B contract changelog (experiment-only, ADR-0094)

Every edit from the arm-A precision fixtures to `eval/contracts.py`, with the real-API reason a careful Creator would
make it. The edits were written from each API's semantics before any arm-B result existed. They were not tuned to
observed failures; the only observations used are the earlier baselines that motivated the experiment. Distractors
keep their arm-A contracts.

| Assistant / Action | Edit | Real-API justification |
| --- | --- | --- |
| dns (Genesis) | States that zones and records are addressed by id, and the list-then-change order. | DNS provider APIs (for example a zone-scoped REST API) address records by zone id and record id only. |
| dns list-zones | Says ids and names are returned. | The zones endpoint returns both. |
| dns list-records | Documents that every filter is exact, that `name` is a record name and never an address, and adds an exact `content` filter (for example an IPv4 address). | DNS record list endpoints offer exact `name`, `type`, and `content` filters; content search is how a caller finds which records point to an address. |
| dns list-records | Declares `name` a hostname kind (arm-C metadata) and `name`/`content` as filters. | A record name is a hostname; an address in a name filter can never match. |
| dns get-record (new) | Get one record by id. | DNS APIs expose a record detail endpoint. |
| dns create-record | Returns `created: false` and the existing record instead of failing when the name and type already exist, and points to update-record. | The provider rejects a duplicate with a typed error; a careful Creator maps it to a result that carries the existing record so the caller can update it. |
| dns update/delete-record | Describes the id-based addressing, with example values. | Same id-only addressing as the API. |
| tasks (Genesis) | States the task fields and that a task is found by title words before acting. | To-do APIs return ids, titles, labels, and completion status. |
| tasks list-tasks | Adds a case-insensitive `query` over title words; documents that `tag` must be an existing tag, not a title word. | To-do APIs offer full-text search and label filters with exact label names. |
| tasks get-task (new) | Get one task by id. | To-do APIs expose a task detail endpoint. |
| tasks create-task | Says due date and tags are set only when the user gives them, with an example title. | Optional API fields should stay unset unless provided. |
| calendar list-events | Documents the returned fields and the YYYY-MM-DD date with an example. | Calendar APIs list a day's events with id, title, start, and duration. |
| calendar create-event | Documents the exact title, the date and 24-hour time formats, and the 60-minute default. | Calendar APIs take an exact title, an ISO date and time, and a default duration. |
| messages find-contact | Documents substring, case-insensitive matching by display name, with an example. | Contact search APIs match parts of display names. |
| messages send-message | Documents that the contact id comes from find-contact. | Messaging APIs address recipients by contact id. |
| research search-web | Advises including the exact service or domain name in the query. | Web search quality depends on the exact entity name. |
| research read-page | Says the URL comes from search results. | A page reader fetches a URL the caller already has. |
| all relevant Actions | Example values in parameter descriptions. | Examples in the published contract show the expected shape, as an API reference does. |
| all Assistants | Declared search terms with per-locale translations (`SEARCH_TERMS`), used only by the arm-D working set. | Creators describe their Assistant in English and the platform translates it into every interface language (ADR-0091). |
