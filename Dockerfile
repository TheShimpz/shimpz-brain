# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# The Shimpz Brain (ADR-0124): one Rust binary, durable chat turns on the vendored Holdpoint, distroless runtime.

# The packages as the dependency build sees them: every manifest and lockfile, and a stub in place of each library and
# binary root, so a source edit leaves the dependency layer unchanged.
FROM docker.io/library/rust:1.99.0-trixie@sha256:6ff07edce8775d0f64be7aba9197229407301bddf2054d62c27b541a6238a181 AS rust-skeleton
WORKDIR /workspace
COPY runtime/Cargo.toml runtime/Cargo.lock ./runtime/
COPY runtime/src ./runtime/src
COPY vendor/holdpoint ./vendor/holdpoint
COPY vendor/strict-json ./vendor/strict-json
COPY protocol/action-schema ./protocol/action-schema
COPY protocol/team/http ./protocol/team/http
RUN find . -type f -regextype posix-extended \
        ! -regex '.*/(Cargo\.toml|Cargo\.lock)' ! -regex '.*/src/(lib|main)\.rs' -delete \
    && find . -depth -type d -empty -delete \
    && for root in $(find . -path '*/src/lib.rs' -o -path '*/src/main.rs'); do \
        printf '//! Dependency stub.\n' > "${root}"; \
        if test "${root##*/}" = main.rs; then printf 'fn main() {}\n' >> "${root}"; fi; \
    done

# The builder's Debian 13 (trixie) glibc matches the distroless cc-debian13 runtime the binary runs on. Every
# dependency is compiled here with the release profile, pinned by runtime/Cargo.lock; the stub packages' artifacts are
# then removed, so the real sources always compile.
FROM docker.io/library/rust:1.99.0-trixie@sha256:6ff07edce8775d0f64be7aba9197229407301bddf2054d62c27b541a6238a181 AS rust-dependencies
WORKDIR /workspace
COPY --from=rust-skeleton /workspace ./
RUN cd runtime \
    && cargo build --release --locked --bin shimpz-brain \
    && cargo clean --release --locked --package brain-runtime --package holdpoint --package strict-json \
        --package action-schema --package team-protocol

FROM rust-dependencies AS rust-build
COPY runtime/Cargo.toml runtime/Cargo.lock ./runtime/
COPY runtime/src ./runtime/src
COPY model_catalog.json ./model_catalog.json
COPY vendor/holdpoint ./vendor/holdpoint
COPY vendor/strict-json ./vendor/strict-json
COPY protocol/action-schema ./protocol/action-schema
COPY protocol/team/http ./protocol/team/http
ARG SOURCE_DATE_EPOCH=0
RUN cd runtime && cargo build --release --locked --bin shimpz-brain \
    && mkdir -p /mounts/run/shimpz-brain-runtime /mounts/var/lib/shimpz-brain-runtime

# One root-owned binary. The two mountpoints keep the ownership and modes fresh volumes copy: Team's token directory
# (uid 10001, the token group 10016, 0750) and the private state directory (uid 10001, 0700). The runtime identity
# can modify neither the binary nor the read-only root filesystem; the healthcheck subcommand needs no shell.
FROM gcr.io/distroless/cc-debian13@sha256:ed7c407fd64eb0af9dddb9456b94cee188a40a7f53cf38c9836e1e9ae14fca02 AS runtime
LABEL org.opencontainers.image.title="shimpz-brain" \
      org.opencontainers.image.description="Provider-neutral Shimpz Brain runtime on Holdpoint"
COPY --link --from=rust-build --chmod=0555 /workspace/runtime/target/release/shimpz-brain /usr/local/bin/shimpz-brain
COPY --link --from=rust-build --chown=10001:10016 --chmod=0750 /mounts/run/shimpz-brain-runtime /run/shimpz-brain-runtime
COPY --link --from=rust-build --chown=10001:10001 --chmod=0700 /mounts/var/lib/shimpz-brain-runtime /var/lib/shimpz-brain-runtime
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=5s --start-interval=250ms --retries=5 \
    CMD ["/usr/local/bin/shimpz-brain", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/shimpz-brain"]
CMD ["serve", "--egress", "shimpz-brain-egress:8888"]
