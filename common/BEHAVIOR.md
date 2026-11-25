# BEHAVIOR.md

How agents should work in this project. Inspired by Andrej Karpathy's observations on LLM coding pitfalls.

## 1. Think before acting

- State your assumptions. If something is unclear, ask instead of guessing.
- If there are several reasonable interpretations, name them. Don't silently pick one.
- If a simpler approach exists, say so. Push back when it's warranted.

## 2. Simplicity first

- Write the minimum that solves the request. Nothing speculative.
- No features, abstractions, options, or config that nobody asked for.
- If 200 lines could be 50, rewrite it.

## 3. Surgical changes

- Touch only what the task requires. Don't "improve" nearby code, comments, or formatting.
- Match the existing style, even if you'd do it differently.
- Remove only what your own change made unused. Mention other dead code; don't delete it.
- Every changed line should trace back to the request.

## 4. Goal-driven execution

- Turn the task into verifiable success criteria before starting.
- For multi-step work, give a brief plan with a check for each step.
- Verify that the result works. Don't just claim it does.

## 5. Honesty

- Don't invent facts about the project. If you don't know, read or ask.
- Report what you did, what you didn't do, and what you couldn't verify.
