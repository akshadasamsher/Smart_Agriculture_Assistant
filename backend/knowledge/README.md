# Knowledge notes

Plain markdown reference notes: `crops/` (22 crop labels) and `diseases/`
(29 disease classes the model predicts). Filenames are the label, lowercased,
with non-alphanumerics replaced by underscores (e.g. `Tomato - Late Blight` ->
`tomato_late_blight.md`).

These are not used by the API any more (they only fed the removed LLM layer).
They are kept in case you want to show them as static guidance in the UI.
The disease notes are general starter notes with no product names or doses -
review them against ICAR / state agricultural university sources before relying on them.
