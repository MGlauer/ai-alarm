# Design

This document describes the design of a prototypical AI-driven security system.

## Components and Agents

This system uses a langgraph-based workflow. The Situation Interpreter (SI) is the LangGraph graph, with one instance (and one checkpointed thread) per situation. The system further consists of one controller, which is plain deterministic code, and multiple agents that operate only upon request. The agents communicate via a pre-defined signal schema defined as Pydantic models. A response of an agent that does not validate against its schema is treated like a response that did not arrive (see the timeout handling below).

The system consists of the following components:

### CCTV/Audio Processor and Filter

Multiple lightweight CCTV cameras and audio sensors record different regions of the surveiled area. It stores the last 5 minutes of video. This processor operates in two possible modes:
* Passive: Each processor analyses a 10-frame window of video from CCTV (~5 seconds) cameras. As a primary filter, it detects unusual changes in the video. This filter should err on the side of caution, as subsequent models will cover the more thorough analyses. Upon detection of an anomaly, it notifies the controller. It passes the triggering sequence of frame and a rough bounding box of the anomalous area.
* Streaming: In addition to the actions of the "Passive"-mode, the processor sends a continuous stream of video to the situation interpreter. Receiving the {stop stream=[stream_id]} signal returns the audio stream to the passive mode.

**Outgoing Signals:**

* {video_event/audio_event start_time="[isotime]" evidence="audio"} - The audio recorder has detected an anomalous sequence of audio.

Accepted events:
* {start stream=[stream_id]} - The recorder should start recording the last minute of video/audio.
* {stop stream=[stream_id]} - The audio recorder should stop recording the last minute of video/audio.
* {get_video/get_audio start_time="[isotime]" end_time="[isotime]"} - The processor should return the video/get_audio between the given timestamps. May return {out_of_scope} if the timestamps are outside the video's time range.

[Implementation note: For the challenge, this component can be a stud that returns a predetermined set of anomalous frame sequences.]


### Person Identificator (Agent)

The person identificator is identifying a person in a given image or video. 

**Incoming Signals:**

* {identify_person video="[video]"} - The person identificator should identify the person in the given video.
* {identify_person image="[image]"} - The person identificator should identify the person in the given image.

**Outgoing Signals:**
* {person_identified person="[person_data]" confidence=[c]} - The person identificator has identified a person in the given video/image. The confidence $c \in [0,1]$ expresses how certain the match is. A match below a pre-configured confidence threshold is treated as *not decidable*.
* {person_not_known} - The person identificator detected a person in the given video/image, but it did not match any known person (*unknown*).
* {unidentifyable cause=[cause]} - The person identificator could not identify a person in the given video/image (e.g. because of a hood, umbrella, or similar obstructions). Cause may be "clothing", "object on person", "external object", "unknown". This is treated as *not decidable*.


### Object detector (Agent)

This agent is responsible for detecting and classifying objects in the video. It should be able to detect multiple objects at once and their possible relations (A man leaving a car, a woman carrying a crowbar, etc). For people, it should also be able to determine the role (e.g. job) of a person. It may also classify an anomaly as a *sensor artefact* (e.g. lens flare, compression glitch, insects on the lens) if it does not show any object.

### Knowledge Base (Graph Database)

A (graph-based) knowledge base should be used to store relevant information:

1. People that the system may recognize, storing name. Each person is represented by one or more images (e.g. uploaded photos or captured by the CCTV cameras). Each person should also be annotated with a familiarity and suspicion score as well as their respective roles (family members, jobs, etc.).
2. A record of all areas of the surveiled area as well of their interconnections.
3. Permission records tracking permission rules for areas (whitelist). These rules should have the shape of simple rules. Natural language examples:
  * Family members are allowed to enter all areas.
  * Delivery people may enter the entryway, but no other areas. However, a dedicated drop-off point may be configured. In that case, one or more paths from the entrance to the drop-off point are allowed.
  * People that are not familiar and have not been invited in/accompanied by a family member in the immediate past are allowed to enter the entryway for a limited amount of time (e.g. to ring the bell) but no other areas.
4. A record of all sensors and their spacial position and orientation
5. A record of all people currently in the surveiled area.

**Note:** The storage and processing of personal data and image data in particular has legal implications under the GDPR, in particular, if the camera's field of view includes public areas. I will not implement this aspect as it seems outside the scope for this challenge, this limitation should be noted and considered in the design of such a system.

### Noise interpreter (Agent)

This agent takes audio data and classifies it as human activity, animal, weather (e.g. wind, rain), technical noise (e.g. a sensor artefact or an electrical hum) or unknown.

### Behavioural interpreter (Agent)

The behavioural interpreter is an agend that analyses a sequence of frames showing one or more people and tries to predict their intentions and interrelations. It should be able to predict the following intents: 
* Delivery
* Pickup
* Ring the bell
* Move to area
* Invite person
* Accompany person
* Health emergency
* Unknown activity
* Suspicious activity

### Weather Forecast Fetch (Agent)

This agent fetches the local weather forecast from a pre-configured weather forecast service. To limit the traffic to outside systems, this agent requests an update at most once an hour and otherwise returns cached data.

### Weather interpreter (Agent)

This agent takes video and audio data and analyzes the weather conditions and annotates them with the following information:
* High wind (with approximate wind speed)
* Snowfall
* Fog
* Rain
* Cloudy
* Sunny
* Unknown weather conditions

### Speaker (Hardware)

Speakers are placed throughout the surveiled area.

**Incoming Signals:**
    {speak text="[text]"} - The speaker should speak the given text.

### Controller (Agent)

This agent is a central controller. The controller mereley has an executive function. It follows the described workflow and may only use LLM components, when indicated. It catches the {event} signals of all sensors and initializes a new Situation Interpreter and passes the signal to it. It receives the {situation_summary} signal from possibly multiple Situation Interpreters and amalgamates them into a single situation summary. It logs the individual situation summaries and the aggregated situation summary as well as the corresponding data.  Whenever the controller triggers a warning or an alarm, the situation and all relevant data is stored in the event log.

An {alarm} signal is triggered if the aggregated situation summary indicates a high suspicion level. Alarms are only ever triggered by the controller through the deterministic rules described here (or by a user elevating a warning); no agent output, in particular no free text, can trigger an alarm directly.

The controller collates all information from all active Situation interpreters, calculates a combined suspicion sore for each detected person. If a person exceeds a pre-configured warning threshold of suspicion, the controller triggers a {warning cause="Suspicious person"} signal. If a person exceeds another threshold of suspicion, the controller triggers an {alarm cause="strong suspicion"} signal. The Controller also compares the area in which a person is detected with the areas in which the person is allowed to enter. If the person is not allowed to enter the area in question, the controller triggers a {warning cause="Unpermitted entry"} signal to the log and a {speak text="your are entering without permission. Please leave the area immediately."} signal is sent. If the entry is prolonged, it will issue an {alarm cause="Unpermitted entry"} signal. Every warning carries the suspicion level of the situation summary it originates from.

Finally, the controller stores the data of all detected people in the knowledge base, alongside their roles and suspicion scores, unless the person was marked as unidentifiable.

If a high-danger animal (e.g. a bear) is detected by some SI, an {alarm [cause]="dangerous animal"} should be triggered under any of the following conditions:
  * There is an open entry point to the house.
  * (Optional; default: False) One or more people are within the surveiled area.
  * (Optional; default: False) The event occurs within a specific time frame.

If none of these three conditions is met, a {warning [cause]="dangerous animal"} is sent instead. If an SI sends the {obscured} signal, the controller queries the weather forecast fetcher for the weather forecast. The results are then evaluated to determine whether the obstruction is plausible (e.g. by heavy fog, snow, rain or an eclipse). It also passes the video data to the behavioural interpreter to check whether the obstruction is plausably caused by human activity of an person flagged as *unsuspicious* or within the scope of their role (e.g. the gardener blocking the view of a camera while trimming the hedges). If the obstruction is not plausible and the obstruction is prolonged, the controller triggers a {warning cause="vision obstructed"} signal. The controller then also queries the respective audio recorder for the audio data and passes the results to the noise interpreter. If the interpretation indicates human activity and no family member is in the area in question, the controller triggers an {alarm cause="obstruction by human activity"} signal.

Each situation and each event is given a unique ID. Alarms, warnings and speaker signals are idempotent: each of them is identified by an idempotency key derived from the situation ID and its cause (e.g. `[situation_id]:[cause]`). Before sending one, the controller checks in a persistent store whether the key has already been used and, if so, does nothing and returns the earlier result. Otherwise, it sends the signal and stores the key. This ensures that retries, a resume from a checkpoint or several SIs reaching the same conclusion do not trigger the same signal twice. Additionally, the system checks, whether a sufficiently similar alarm (same cause, same area) has already been triggered in the immediate past, in order to avoid flooding for distinct situations.

### Situation Interpreter (Agent)

The situation interpreter is controlled by an internal state machine. By default, it is in an `Idle` state, that is only interrupted by an {event} signal from the Controller. It then switches to the `Interpret` analyses the incoming signals. Any subsequent {event} triggers regarding the same region are deferred. During the interpretation, the agend performs the following actions:

The SI passes the information the object detection agent and fans out in parallel to several agents depending on the kind of detected object/event:

* If one or more people are detected. 
  1. The SI queries the `person identificator` for the person's data. If a detected person is known to the system, the SI annotates the bounding boxes of each detected person with the person's name and the person's familiarity, suspicion score and role. The SI then also compares the person's predicted role with the ones in the knowledge base. A notable difference between the annotated roles and predicted ones is considered *suspicious*, wherein the difference in roles influences the intensity of that suspicion. If the `person identifier` is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the person as *not decidable* (cause "unavailable"). A *not decidable* person is not considered *suspicious* by itself, but is treated like a person that is not familiar in the permission checks.
  2. The SI then passes the image and video data to the `behavioural interpreter`. The behavioural interpreter then analyses the data and predicts the person's intentions and interrelations. It compares the predicted intentions with the predicted roles. A mismatch between the predicted intentions and predicted roles is considered *suspicious*, wherein the degree of mismatch ($d \in [0,1]$) between both characteristics influence the intensity of that suspicion. The behavioural interpreter may also detect possible health emergencies, which are also annotated. If the behavioural interpreter is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the behaviour as *unknown*.

* If an animal is detected, the SI annotates the potential dangerousness with a pre-defined danger score depending on its kind (1 for a bear, 0.8 for a boar, 0.6 for an unknown large dog, 0.4 for an unknown small dog, etc.). If the animal is not known to the system, it is annotated as *unknown*, which is by default not considered *dangerous*.

* If a weather event has been detected, the SI queries the weather forecast fetcher for the weather forecast and compares the results. If there is a significant mismatch between these results, the situation is considered suspicious and the weather event anntotated accordingly. If either the weather forecast fetcher or the weather interpreter is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the weather event as *unknown*, which is by default not considered *suspicious*.

* If the anomaly was classified as a *sensor artefact* (by the object detector or, for audio, the noise interpreter), the SI annotates it accordingly. It has a threat score of 0 and is only logged.

* If the vision of the CCTV systems is obscured, the SI returns an {obscured} signal. 

The SI merges the respective responses and compiles them into a `situation summary`. It sends a {situation_summary} signal to the controller, which contains a summary of the situation and a separate threat assessment assigned with a score $[0,1]$. The score is the minimum of the scores of the individual objects/events. The SI then moves to the `Observe` state, in which it reevaluates the situation at fixed intervalls and on subsequent {event} signals by re-entering the `Interpret` state. Once the re-evaluation determines that the situation is resolved, it sends a {situation_resolved} signal to the controller and moves to the `Idle` state.

The state of each situation is checkpointed by the workflow after each step, keyed by the situation ID (the thread ID). After a crash or restart, the SI resumes from the last checkpoint with the same situation ID. Steps that are repeated after a resume only re-run analyses; side effects are protected by the idempotency keys described above.

### Communication Unit (Agent)

Upon receiving a {warning cause="[cause]"} the communication sends a text notification to all registered primary contacts and also to all contacts within the surveiled area. A warning with a suspicion level from over 0.75 is displayed as an orange warning and those below as a yellow warning. During the night, this threshold is lowered to 0.5, as users are less likely to respond at night and unanswered orange warnings are elevated. A user then have the option to elevate the warning to an alarm, by clicking a button in the linked app, which sends an {elevate warning_id=[warning_id]} signal. While waiting for the answer, the situation is paused (an interrupt of the workflow) and resumed under the same situation ID once an answer or the timeout arrives. If a warning is not answered within a pre-defined time frame either by an dismiss or an elevate, the system falls back and dismisses yellow warnings and elevates orange warnings automatically.

Each kind of alarm and warning has a pre-defined template for outside communication, which is completed with information from the annotated data in the signal. No messages for outside communication are autogenerated.