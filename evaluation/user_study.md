# User study (test 3, with people)

The automated part of test 3 checks that the whole pipeline works. This study checks that the **experience** is
fluid for real users, as the evaluation plan in the proposal asks. It takes about 15 minutes per person.

## Setup

* **Participants:** 5–8 classmates. Note for each whether they have musical training (yes/no) and whether they have
  used a DAW before (yes/no). Musical experience is not required: the proposal's claim is that no music theory is needed.
* **Where:** the live site (open it about 2 minutes early so the free Render server has woken up) or a local run.
* **Files to hand out:** `evaluation/data/melodies/ode_to_joy.mid` (made by `run_evaluation.py`).
* **Facilitator:** read the task, start a timer, don't help unless the person has been stuck for 2 minutes
  (then mark the task "helped"). Note anything they say out loud.
* **Consent:** tell participants their answers are anonymous and used only for the course report. Record a
  participant number (P1, P2 …), not names.

## Tasks

| # | Task (read this to the participant) | Success means |
|---|---|---|
| T1 | "Use the **From prompt** tab to make a short piece that sounds like a calm rainy evening. Listen to it." | Generated and pressed Play |
| T2 | "Upload `ode_to_joy.mid` in **From MIDI** and add a bass line and drums to it. Listen to it." | Generated with the melody kept |
| T3 | "Read the **Critic** for your result and follow one of its suggestions." | Used a critic button, or changed a setting it named |
| T4 | "Add one more instrument part to the result **without changing what is already there**." | Used Build on this |
| T5 | "Download the parts as separate files." | Downloaded the stems .zip |

For each task record: **completed** (yes / helped / no) and **time** in seconds.

## Questionnaire (after the tasks)

**A. System Usability Scale (SUS)**, 1 = strongly disagree … 5 = strongly agree:

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

**B. MIDI Magic questions**, same 1–5 scale:

* Q1. The music matched the mood I asked for.
* Q2. The app understood my text prompt.
* Q3. The parts it added fit the melody I uploaded.
* Q4. The Critic's feedback was useful.
* Q5. Having separate, editable parts (stems) is useful to me.
* Q6. Compared with a tool that only gives a finished audio file, I had more control here.

**C. Open questions:** What was the most confusing moment? What would you add or change?

## Recording results

Fill in `evaluation/user_study_results.csv` (one row per participant; leave a cell empty if not asked), then run:

```bash
python evaluation/analyze_user_study.py
```

It writes `evaluation/results/user_study.md` with the SUS score (0–100; about 68 is average), task completion rates
and times, and the mean and spread of every question, ready for the report.
