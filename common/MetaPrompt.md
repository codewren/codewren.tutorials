# Prompt to regenerate these docs

Copy the prompt below into a coding agent, run from the project root, to regenerate
docs

```text
Write a tutorial, as Markdown files under `docs/`, that explains [ how a decoder-only
Transformer LLM works during INFERENCE: from raw text in to the next token out.]. The 
tutorial is chapter based.

Audience: engineers new to the area. Tone: concise, tutorial-style.

FILES
- File name pattern: ch-N-title.md (or .pdf, .ppt depending on the file type)


GLOBAL RULES
- Chapter 0 should be the overview. It should include the linked chapter table, the Notation section as other overview contents.
- Diagrams: Mermaid, ALWAYS horizontal (`flowchart LR`, and `direction LR` inside
  subgraphs). ASCII diagrams are allowed for matrices/shapes.
- Use ONLY the symbols defined in the Chapter 0 Notation table, with one meaning each.
- Chapter layout: H1 "Chapter N — Title", short intro, diagram(s), formulas , running-example numbers, a tiny hand-checkable worked example, a short NumPy
  snippetb(optional), "Key takeaways" bullets, then a footer with ← previous / next → chapter links
  (Last Chapter  links forward back to Ch 0).
- Verify ALL arithmetic (run it) before writing it down.
- Each chapter should include the reference publised paper or text book for the source of information. Keep the most important 10 reference item. 
- The reference papers should be published either in a top conference in the field or in a high-reputation journal. 
- The reference blog should be from a well-known company or organization in the field.
```

# Prompt to generate the re-gen Prompt

```
Summary the generated doc and write a prompt so that i can regenerate it again later.
```
