# AI Usage Disclosure

## Tools used

| Tool | Used for | Approx. share |
|---|---|---|
| Claude (Anthropic) | Code review and bug-fix suggestions in Part B; explaining results; drafting the README sections, the notebook findings text and the Part C memo; drafting a first pass of the 50 complaint labels | ~35% of Part B code, ~25% of Part A code, and the first draft of the Part C memo |
| OpenAI gpt-4o-mini | The triage model itself (runtime component, not coding help) | n/a |

## What I did myself
- **Labels:** Claude drafted a first pass of the 50 complaint labels. I reviewed and corrected them. The number of `requires_human = true` labels went from 13 to 17 after my review.
- **Code changes:** I applied the Part B improvements (hardship keywords in the safety regex, category definitions in the prompt) and reran the whole project, including the full 300-complaint evaluation.
- **Writing:** Claude drafted the README sections, the notebook findings and the Part C memo. I edited them and checked the numbers against the notebook outputs.

## Honest limits
- Claude's first-pass labels may have influenced my own labelling, so the Part B scores may be slightly optimistic.
- I tuned the Part B prompt and regex on the same 50 labels I report on, so run 2 is optimistic. This is stated in the README and the memo.
- The Part C memo started as a Claude draft. I am ready to explain and defend every claim in it.