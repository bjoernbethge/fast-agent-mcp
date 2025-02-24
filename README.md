## FastAgent

<p align="center">
<a href="https://pypi.org/project/fast-agent-mcp/"><img src="https://img.shields.io/pypi/v/fast-agent-mcp?color=%2334D058&label=pypi" /></a>
<a href="https://github.com/evalstate/fast-agent/issues"><img src="https://img.shields.io/github/issues-raw/evalstate/fast-agent" /></a>
<a href="https://lmai.link/discord/mcp-agent"><img src="https://shields.io/discord/1089284610329952357" alt="discord" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/fast-agent-mcp?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/evalstate/fast-agent-mcp/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/fast-agent-mcp" /></a>
</p>

## Overview

**`fast-agent`** lets you define, test and interact with agents, tools and workflows in minutes.

The simple declarative syntax lets you concentrate on the prompts, MCP Servers and compositions to build effective agents.

Quickly compare how different models, combine

### Get started:

Install the [uv package manager](https://docs.astral.sh/uv/).

```bash
uv pip install fast-agent-mcp       # install fast-agent
fast-agent setup                    # create an example agent and config files
uv run agent.py                     # run your first agent
uv run agent.py --model=o3-mini.low # specify a model
fast-agent bootstrap workflow       # create "building effective agents" examples
```

`fast-agent bootstrap workflow` - generate example agents and workflows demonstrating each of the patterns from Anthropic's "[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)" paper.

`fast-agent bootstrap` -

It's built on top of [mcp-agent](todo).

### llmindset.co.uk fork:

- "FastAgent" style prototyping, with per-agent models
- Warm-up / Post-Workflow Agent Interactions
- Quick Setup
- Interactive Prompt Mode
- Simple Model Selection with aliases
- User/Assistant and Tool Call message display
- MCP Sever Environment Variable support
- MCP Roots support
- Comprehensive Progress display
- JSONL file logging with secret revokation
- OpenAI o1/o3-mini support with reasoning level
- Enhanced Human Input Messaging and Handling
- Declarative workflows

## Get Started

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python projects:

## Table of Contents

We welcome any and all kinds of contributions. Please see the [CONTRIBUTING guidelines](./CONTRIBUTING.md) to get started.

```

```
