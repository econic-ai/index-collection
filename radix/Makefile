SHELL := /bin/zsh

.PHONY: help build run test bench bench-quick extract destroy check fmt clean

OPS ?=
LFS ?=
TAG ?=
CARGO_TARGET_DIR := target
BENCH_ARGS ?=
make extract IMPL=mo_hashbrown ?= m0_hashbrown
ifeq ($(IMPL),m0_hashbrown)
BENCH_FEATURE := bench-m0-hashbrown
else ifeq ($(IMPL),radix_tree)
BENCH_FEATURE := bench-radix-tree
else ifeq ($(IMPL),all)
BENCH_FEATURE := bench-m0-hashbrown,bench-radix-tree
else
$(error Unsupported IMPL '$(IMPL)'. Use IMPL=m0_hashbrown, IMPL=radix_tree, or IMPL=all)
endif

BENCH_ENV := CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)
ifneq ($(strip $(OPS)),)
BENCH_ENV += OPS=$(OPS)
endif
ifneq ($(strip $(LFS)),)
BENCH_ENV += LFS=$(LFS)
endif

help:
	@echo "Radix common commands:"
	@echo "  make build       Build radix package"
	@echo "  make run         Run src/main.rs"
	@echo "  make test        Run all tests"
	@echo "  make bench       Run Criterion benchmarks (IMPL selects compile-time index)"
	@echo "  make bench-quick Compile benchmark target only"
	@echo "  make extract     Export Criterion summaries to CSV (requires IMPL, optional TAG)"
	@echo "  make destroy     Remove CSV + Criterion data for IMPL (m0_hashbrown|radix_tree|all)"
	@echo "  make check       Run cargo check"
	@echo "  make fmt         Run rustfmt"
	@echo "  make clean       Remove target artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make bench IMPL=m0_hashbrown OPS=lookup_hit,lookup_miss LFS=0.5,0.9 BENCH_ARGS=--quick"
	@echo "  make bench IMPL=radix_tree OPS=lookup_hit LFS=50,90 BENCH_ARGS=--quick"
	@echo "  make bench IMPL=all OPS=lookup_hit LFS=0.5 BENCH_ARGS=--quick"
	@echo "  make extract IMPL=m0_hashbrown TAG=first_pass"

build:
	cargo build

run:
	cargo run

test:
	cargo test

bench:
	echo $(BENCH_ENV)
	$(BENCH_ENV) cargo bench --bench m0_bench --no-default-features --features $(BENCH_FEATURE) -- $(BENCH_ARGS)

bench-quick:
	CARGO_TARGET_DIR=$(CARGO_TARGET_DIR) cargo bench --bench m0_bench --no-default-features --features $(BENCH_FEATURE) --no-run

extract:
	@test -n "$(IMPL)" || (echo "IMPL is required, e.g. make extract IMPL=m0_hashbrown [TAG=run1]" && exit 1)
	python3 analysis/export_criterion.py --impl "$(IMPL)" --tag "$(TAG)"

destroy:
	@test -n "$(IMPL)" || (echo "IMPL is required, e.g. make destroy IMPL=m0_hashbrown or IMPL=all" && exit 1)
ifeq ($(IMPL),all)
	@echo "Destroying CSV + Criterion data for ALL implementations..."
	rm -f analysis/data/m0_hashbrown.csv analysis/data/radix_tree.csv
	rm -rf target/criterion/impl=m0_hashbrown_op=* target/criterion/impl=radix_tree_op=*
else
	@echo "Destroying CSV + Criterion data for IMPL=$(IMPL)..."
	rm -f analysis/data/$(IMPL).csv
	rm -rf target/criterion/impl=$(IMPL)_op=*
endif
	@echo "Done. Run 'make bench' then 'make extract' to regenerate."

check:
	cargo check

fmt:
	cargo fmt

clean:
	cargo clean
