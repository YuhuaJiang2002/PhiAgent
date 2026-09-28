# Robot video skill
## Workflow
Preserve the exact requested action, named robot arm, object and endpoint.
Ground the prompt only in the literal task and visible initial image.
Describe one continuous eight-second action while preserving the camera and initial scene.
Use a concise positive prompt of at most 180 words.
## Error Avoidance
Do not invent extra actions: picking up ends in a stable hold; placing ends with the object resting; do not describe arm retraction or movement after the object is placed or held.
Do not assert an invisible grasp, support state, future outcome, or the origin of an object already held.
Do not mention judges, scores or unavailable ground truth.
