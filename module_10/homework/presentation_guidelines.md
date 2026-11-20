# Final Presentation Guidelines

**For:** Lecture 40, two sessions: Wednesday December 2 and Friday December 4, 2026 (Week 14)
**Format:** 4-minute talk + 1–2 minutes Q&A per student
**Grading:** 10 of the final project's 100 points (the final project is 40% of the course grade)

---

## What we want to see

By the end of your 4 minutes, the audience should know:

1. **What problem you solved.** One sentence.
2. **What method you used.** PINN / data-driven constitutive / open option / hybrid.
3. **What the result was.** Quantitative metric + figure.
4. **What you learned.** A specific limitation, surprise, or insight.

---

## Suggested slide format

4–5 slides for a 4-minute talk; Slide 6 only if you have time. **Less is more.** A dense slide that takes 30 seconds to digest costs you 30 seconds of your talk.

### Slide 1 — Title

- Project title (1-2 lines).
- Your name and a thank-you to the instructor.

### Slide 2 — Problem

- The mechanics problem in 1 picture (geometry + BC + what you're trying to predict).
- 1 sentence: "We want to predict X given Y."
- 1 sentence: "This matters because Z."

### Slide 3 — Method

- Your network architecture in 1 figure.
- 1 sentence: "We chose this architecture because W."
- 1 sentence: data source and amount.

### Slide 4 — Results (the centerpiece)

- The single most compelling figure: parity plot, comparison vs reference, or learning curve.
- 1 quantitative number: "Our MAE was X, beating the baseline by Y%."
- Be honest about the metric.

### Slide 5 — Limitations + Future Work

- 1 limitation you discovered during the project. Honest.
- 1 thing you'd extend if you had another month.

### Slide 6 — Surprises (optional, encouraged)

- 1 thing that surprised you. The audience loves these.
- Could be: a method that didn't work as expected; an unexpected hyperparameter sensitivity; a piece of physics that the model uncovered.

---

## Tips

**Slide design.**

- Big fonts (24+ point). The back of the room can't read 12-point text.
- One figure per slide. Don't crowd.
- Color sparingly. Black-on-white reads cleanly in any projection.

**Verbal delivery.**

- Speak slower than you think you should. Nerves make you speed up; counter it.
- Don't read slides. Glance + look at the audience + speak.
- Pause for emphasis after key numbers.

**Live demos.**

- Only if 100% robust. A failed live demo eats your time and confidence.
- Otherwise: screenshots or 15-second pre-recorded video.

**Q&A.**

- "I don't know" is an acceptable answer. Don't pretend.
- "Good question" is a useful stalling phrase while you think.
- Aim to answer in 30 seconds; let the asker follow up if they want more.

---

## Logistics

**Send your slides PDF to the instructor by 11:59 PM the night before your session**: Tuesday December 1 if you present Wednesday December 2, Thursday December 3 if you present Friday December 4. Subject line: `[ML4SM Final] <your name>`.

All slides will be pre-loaded on the instructor's laptop. Switching between presenters' laptops wastes time we don't have.

**Bring backups.** USB stick or cloud-storage link with your PDF in case the email version got corrupted.

**Arrive 10 minutes early.** Each session starts at the regular class time (1:10 PM); the instructor will project the first set of slides at that moment. About half the class presents in each session, so the timing is tight: 4 minutes means 4 minutes.

---

## Order

The session assignment (Wednesday December 2 or Friday December 4) and the speaking order will be posted on Canvas by **Monday November 23**, before Thanksgiving, so you know which slide deadline (Tuesday December 1 or Thursday December 3) applies to you.

---

## What the audience will be looking for

Your peers will be asked to ask questions. Common question types they'll lean toward:

- **Method choice.** "Why this architecture and not [other architecture from a different module]?"
- **Comparison.** "How does this compare to the baseline you had in Lab N?"
- **Limitation depth.** "If you had another semester, what would you change?"
- **Generalization.** "Would this work on a different material / geometry / problem?"

The instructor will likely ask one of:

- **The honesty question.** "What didn't work that you wanted to work?"
- **The future-work question.** "What's the natural next step?"
- **The connection question.** "How does this fit into the broader ML4Mechanics literature?"

You don't need rehearsed answers, but having thought about each helps.

---

## Grading rubric (presentation component, 10 of the final project's 100 points)

| Item | Points |
|------|--------|
| Slides clear, on time | 3 |
| Methodology explained | 2 |
| Results presented with appropriate metric | 2 |
| Limitations addressed honestly | 1 |
| Q&A engagement (responsive, honest) | 2 |

The other 90 points come from the code + report (graded separately).

---

## A friendly suggestion

The presentation is not just for the grade. It's for **practice presenting your own research**. Most graduate-school and industry-job interviews include a research talk like this. You will give a short summary of your project hundreds of times in your career. This is your first try in this format. Treat it as a rehearsal.

Most importantly: **be excited about what you did**. You worked on this project for half a semester. Show the audience why it was interesting.

Good luck.
