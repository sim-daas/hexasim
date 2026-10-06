# Minor project progress talk: speaking notes

Deck: `Progress_PPT_minor_project.pptx` (18 slides). Notes below match the slide notes in the file.

## Slide 1: Title

Introduce the project and the team. Say this is a progress review: the walking policy is trained and tested in simulation, and the next step is transfer to the real robot. Fill in names, supervisor and date.

## Slide 2: Content

Walk through the nine sections. The talk is about 15 minutes; the results section is the longest.

## Slide 3: Introduction

The robot is a six-legged PhantomX Mark IV with 18 servo-driven joints. The aim is a controller that uses only its own joint readings and IMU, with no camera and no simulator-only signals. Training is in NVIDIA Isaac Lab with PPO. Be clear that the work is in simulation so far; hardware transfer is next.

## Slide 4: Motivation: Issues and Challenges

Explain the sim-to-real gap in plain words: a policy that is perfect in a clean simulator often fails on a real robot. Name the three sources: the simulator is too clean, real servos are imperfect, and the terrain varies. Last bullet sets up the problem we solved.

## Slide 5: Literature Review

Group the literature into four approaches, not paper by paper. Explain where each falls short for a blind, deployable walker. Replace the placeholder with the cited papers before presenting; the notes file lists what to add.

## Slide 6: Problem Definition

This is the problem in one sentence, then the three constraints. Stress that the robot must not see anything the real robot cannot sense. The three bullets map to the next sections.

## Slide 7: Research Gap

Three gaps, each tied to a result later. Gap 1 motivates the servo model and terrain curriculum. Gap 2 motivates removing ground-truth velocity. Gap 3 is the problem we fixed, covered in detail later.

## Slide 8: Objectives

Four objectives, matching the constraints. The fourth is about how we report results: fixed seeds, held-out terrain and several training seeds, so the numbers can be reproduced.

## Slide 9: Proposed Method: Control Loop

Read the diagram left to right. The command and gait clock go into the policy. The policy outputs 18 joint targets. The servo model and the robot respond, and the sensor readings feed back. The reward and critic exist only in training and are not on the robot.

## Slide 10: How We Built the Policy: Four Stages

Each stage adds one thing and must pass its gate before the next starts. Stress the order: the simulator is corrected first, then the servos, then commands and terrain, then the spatial material patches.

## Slide 11: Training Terrain and Domain Randomisation

This is the main sim-to-real slide. Each range brackets something the real robot may differ on: friction, payload, servo strength, delay and sensor noise. The held-out levels 15–20 are never used in training. Be honest that the ranges are provisional until the robot is measured on the bench.

## Slide 12: Problem We Solved: Legs Passing Through Each Other

The problem: an earlier policy's legs overlapped in simulation, with the worst gap at minus 6.3 cm, and about 22 % of steps were under the 2 cm safety margin. The root cause is that self-collision was off, so nothing penalised it. The chart shows the gap after the fix: positive means a clear gap.

## Slide 13: Problem We Solved: The Fix

Three changes: a leg-proximity penalty with a 2 cm margin, coordinated swing so the middle and back legs lift together and land at the rest reach, and retraining from scratch. The economics check: walking still beats standing and falling. The lesson for the audience: stride metrics alone do not catch unsafe gaits.

## Slide 14: Results: Training-Range Gates (Stage S0b)

Zero falls in 17,408 evaluation episodes. Stride error is within plus or minus 1.4 % at both speeds, and turning is close to the commanded 0.3 rad/s in both directions. All 108 training terrain cells pass. On the held-out levels 15–20, 50 of 72 cells pass, with no falls; be open that these harder cells are weaker.

## Slide 15: Results: Blind Terrain Policy (Stage S1, Five Seeds)

Five training seeds, all passing. Falls are zero in 21,760 episodes. The per-seed chart shows the spread: the difference between seeds is small compared with the gate margins. The velocity error is about 0.03 m/s on rigid ground and about 0.06 m/s on slippery ground. This is the policy that goes forward to hardware.

## Slide 16: Plan of Action for the Project

Left is done. Right is the plan. Be clear that the headroom test is the decision point: it decides whether the camera-based stages go ahead. Phase 3 is only mentioned here, as a design-stage model that will be compared under the same protocol.

## Slide 17: Conclusion and Future Directions

Close on three points: a blind walker that passes every gate across five seeds, a gait fix that came from a physical check rather than the speed metric, and the next steps. Stop before the headroom result, which is not part of this review.

## Slide 18: Thank You

Thank the panel and open for questions.

