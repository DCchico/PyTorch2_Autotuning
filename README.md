
# Rethinking Autotuning in ML Compilers

This repository hosts an ongoing research project investigating the **reliability and efficiency of measurement-driven optimization decisions in modern ML compilers**, with an initial focus on PyTorch 2 and TorchInductor.

## Motivation

Modern ML compilers use autotuning to select among alternative implementations and kernel configurations. These decisions typically rely on candidate-level performance measurements, which may not accurately reflect the relative performance of implementations during final compiled execution.

Our previous research, **PerformanceDNA**, identified cases where TorchInductor's autotuning benchmark rankings disagreed with actual compiled-inference performance.

These observations motivate a broader question:

> Can we improve how ML compilers collect, interpret, and reuse performance evidence to make better optimization decisions at lower tuning cost?

Rather than only characterizing existing compiler decisions, this project explores opportunities to improve the underlying autotuning and implementation-selection mechanisms.

## Research Directions

We investigate three related challenges:

### 1. Measurement and Ranking Fidelity

Understand when and why autotuning measurements fail to preserve implementation rankings under actual execution.

- Characterize ranking discrepancies across optimization decision families.
- Identify their underlying execution and measurement mechanisms.
- Determine when inexpensive measurements are sufficiently reliable.

### 2. Cost-Efficient Autotuning

Explore how to make reliable optimization decisions without exhaustively measuring every implementation under its full execution context.

- Quantify the tradeoff between measurement fidelity and tuning overhead.
- Investigate selective verification of promising candidates.
- Minimize additional measurement effort while preserving selection quality.

### 3. Measurement Evidence Transferability

Investigate whether performance evidence collected under one workload or execution context can be reused in another.

- Identify conditions under which implementation rankings remain stable.
- Explore interpretable corrections for predictable context-dependent changes.
- Determine when additional measurement or re-verification is necessary.

These directions are exploratory and may evolve as experimental evidence develops.

## Research Scope

Our initial investigation focuses on PyTorch 2 and TorchInductor, including:

- GEMM implementation selection and Triton kernel configurations.
- Convolution and reduction autotuning.
- Candidate-level versus compiled-execution measurements.
- Normal kernel launches, CUDA Graph replay, and other relevant execution contexts.
- Workload characteristics, hardware differences, and compilation/tuning costs.

We aim to evaluate existing autotuning mechanisms, understand their limitations, and investigate potential compiler-level improvements.

## Current Status

**Early-stage research and experimental characterization.**

Current work focuses on:

- Source-code analysis of existing autotuning mechanisms.
- Reproducing benchmark-to-execution ranking discrepancies.
- Evaluating independent autotuning decision families.
- Investigating measurement fidelity and tuning-cost tradeoffs.
- Studying related academic and industrial systems.

The eventual system architecture and research contributions have not yet been finalized. All proposed mechanisms remain subject to experimental validation.

## Background

This project builds on our earlier work:

**PerformanceDNA: An Experimental Framework for Systematically Studying and Improving ML Compiler Optimization Decisions**

It also considers related research in compiler autotuning, performance modeling, adaptive measurement, GPU execution, and deployment-aware optimization.
