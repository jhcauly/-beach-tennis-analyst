# Ball, strokes, tactical landing and review clips

## Product outputs

The analysis must generate two synchronized videos:

1. `annotated_match.mp4`: original video with athlete IDs, ball trail, stroke label, landing marker and rally state.
2. `movement_2d.mp4`: top-down court animation with the four athletes, pair centroids, ball trajectory and event markers.

Both videos must use the source FPS and frame timestamps so that every player and ball event can be traced back to the same moment.

## Event pipeline

```text
video frames
-> player tracking and court projection
-> ball detection and short-track association
-> player-ball contact candidate
-> stroke classification
-> ball flight and landing estimation
-> rally segmentation
-> shot outcome classification
-> player shot report
-> automatic review clips
```

## Shot types

Initial classes:

- serve
- smash
- volley
- forehand
- backhand
- lob
- unknown

The system must keep `unknown` whenever confidence is insufficient. Silent confident guessing is not acceptable.

## Outcomes

- winner
- defended
- out
- net
- forced error
- unforced error
- in play
- unknown

A shot marked `defended` means the opponent touched or returned it and the point continued. A `winner` requires no valid opponent return before the rally ends.

## Landing map

Every reliable landing on the opponent half is projected to metric court coordinates and accumulated in a 3x3 tactical grid. Reports must provide:

- total landings by cell
- percentage by cell
- winners by cell
- defended balls by cell
- errors by stroke type
- most successful target zone
- most defended target zone

## Serve and smash report

For each athlete:

- serve attempts
- valid serves
- serve errors into net
- serve errors out
- smash attempts
- smash winners
- smash defended
- smash errors into net
- smash errors out
- success percentage with explicit confidence coverage

## Review clips

The report must create a clip index for actions useful to coaching, especially:

- opponent defended the ball
- ball went out
- ball hit the net
- forced error
- unforced error
- low-confidence classification requiring human confirmation

Default clip window: two seconds before contact and 2.5 seconds after contact. The final renderer may extend the end until the rally outcome becomes visible.

## Confidence gates

Ball and stroke statistics must not be presented as exact when detection coverage is poor. Every report must contain:

- percentage of frames with reliable ball detection
- percentage of shots classified
- percentage of landing points projected
- number of events requiring review

## Development order

1. reliable ball detector and tracker
2. contact detection
3. rally segmentation
4. landing estimation
5. basic outcomes: in play, defended, out, net
6. stroke classifier: serve and smash first
7. synchronized 2D renderer
8. review clip exporter
9. tactical report integration
