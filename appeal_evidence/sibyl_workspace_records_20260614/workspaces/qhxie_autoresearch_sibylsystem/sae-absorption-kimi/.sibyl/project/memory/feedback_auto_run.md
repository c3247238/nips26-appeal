---
name: feedback_auto_run
description: User confirms fully autonomous mode - no questions needed
type: feedback
---

## Rule: Fully Autonomous Operation

**Rule**: The Sibyl system must run fully autonomously without asking the user for decisions or opinions.

**Why**: The user explicitly stated: "继续，我记得这个项目是全自动的不需要问我的意见" (Continue, I remember this project is fully automatic and doesn't need to ask for my opinion).

**How to apply**: 
- Do not ask the user for decisions on research direction, experiment design, or paper framing
- Make all decisions independently based on the system's own judgment
- Proceed with PIVOT, PROCEED, or REFINE decisions without user confirmation
- Only stop if the user explicitly executes `/sibyl-research:stop`
- If uncertain, default to continuing rather than asking
