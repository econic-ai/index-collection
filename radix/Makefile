SHELL := /bin/zsh

.PHONY: help build run test bench bench-quick extract destroy check fmt clean

OPS ?=
OP ?=
LFS ?=
TAG ?=
BENCH ?= m0
IMPL ?= hashbrown
GPU32 ?= 0
CARGO_TARGET_DIR := target
BENCH_ARGS ?=

ifeq ($(IMPL),hashbrown)
BENCH_FEATURE_BASE := bench-hashbrown
else ifeq ($(IMPL),radix_tree)
BENCH_FEATURE_BASE := bench-radix-tree
else ifeq ($(IMPL),all)
BENCH_FEATURE_BASE := bench-hashbrown,bench-radix-tree
else
$(error Unsupported IMPL '$(IMPL)'. Use IMPL=hashbrown, IMPL=radix_tree, or IMPL=all)
endif

BENCH_FEATURE ?= $(BENCH_FEATURE_BASE)
ifeq ($(GPU32),1)
BENCH_FEATURE := $(BENCH_FEATURE),gpu32
endif

# Resolve bench binary name from BENCH parameter.
ifeq ($(BENCH),m0)
BENCH_BIN := m0_bench
else ifeq ($(BENCH),m1)
BENCH_BIN := m1_bench
else
$(error Unsupported BENCH '$(BENCH)'. Use BENCH=m0 or BENCH=m1)
endif

# Criterion impl name: matches what the bench binary emits in group IDs.
# gpu32 appends _gpu to the impl name in the m1 bench.
CRITERION_IMPL := $(IMPL)
ifeq ($(GPU32),1)
CRITERION_IMPL := $(IMPL)_gpu
endif
CSV_NAME := $(BENCH)_$(CRITERION_IMPL)

BENCH_ENV := CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)
# Back-compat alias: allow OP=iter as shorthand for OPS=iter.
ifeq ($(strip $(OPS)),)
ifneq ($(strip $(OP)),)
OPS := $(OP)
endif
endif
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
	@echo "  make bench       Run Criterion benchmarks"
	@echo "  make bench-quick Same as bench but with --quick"
	@echo "  make extract     Export Criterion summaries to CSV"
	@echo "  make destroy     Remove Criterion data for IMPL"
	@echo "  make check       Run cargo check"
	@echo "  make fmt         Run rustfmt"
	@echo "  make clean       Remove target artifacts"
	@echo ""
	@echo "Parameters:"
	@echo "  BENCH=m0|m1       Select benchmark binary (default: m0)"
	@echo "  IMPL=hashbrown|radix_tree|all  Select implementation (default: hashbrown)"
	@echo "  GPU32=0|1         Enable gpu32 feature (default: 0)"
	@echo "  OPS=...           Comma-separated ops filter"
	@echo "  LFS=...           Comma-separated load factor filter"
	@echo "  TAG=...           Tag for CSV export"
	@echo ""
	@echo "Examples:"
	@echo "  make bench IMPL=hashbrown OPS=lookup_hit,lookup_miss LFS=0.5,0.9 BENCH_ARGS=--quick"
	@echo "  make bench IMPL=radix_tree OPS=lookup_hit LFS=50,90 BENCH_ARGS=--quick"
	@echo "  make bench BENCH=m1 IMPL=radix_tree                    # CPU iter across sizes"
	@echo "  make bench BENCH=m1 IMPL=radix_tree GPU32=1             # GPU iter across sizes"
	@echo "  make extract IMPL=radix_tree                            # -> m0_radix_tree.csv"
	@echo "  make extract BENCH=m1 IMPL=radix_tree                   # -> m1_radix_tree.csv"
	@echo "  make extract BENCH=m1 IMPL=radix_tree GPU32=1           # -> m1_radix_tree_gpu.csv"

build:
	cargo build

run:
	cargo run

test:
	cargo test

bench:
	$(BENCH_ENV) cargo bench --bench $(BENCH_BIN) --no-default-features --features $(BENCH_FEATURE) -- $(BENCH_ARGS)

bench-quick:
	$(BENCH_ENV) cargo bench --bench $(BENCH_BIN) --no-default-features --features $(BENCH_FEATURE) -- $(BENCH_ARGS) --quick

extract:
	@test -n "$(IMPL)" || (echo "IMPL is required" && exit 1)
	python3 analysis/export_criterion.py --impl "$(CRITERION_IMPL)" --csv-name "$(CSV_NAME)" --tag "$(TAG)" $(if $(strip $(OPS)),--op "$(OPS)")

destroy:
	@test -n "$(IMPL)" || (echo "IMPL is required" && exit 1)
ifeq ($(IMPL),all)
	@echo "Destroying Criterion data for ALL implementations..."
	rm -rf target/criterion/impl=hashbrown_op=* target/criterion/impl=radix_tree_op=*
else
	@echo "Destroying Criterion data for IMPL=$(IMPL)..."
	rm -rf target/criterion/impl=$(IMPL)_op=*
endif
	@echo "Done. Run 'make bench' then 'make extract' to regenerate."

check:
	cargo check

fmt:
	cargo fmt

clean:
	cargo clean
