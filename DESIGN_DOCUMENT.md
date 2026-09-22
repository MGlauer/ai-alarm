# Design

This document describes the design of a prototypical AI-driven security system. It is based on the Flask framework and uses the LangGraph library for the workflow and a sqlite database for the persistent storage (This is for the challenge only, in production, a more robust database should be used).

## Workflow

This system uses a langgraph-based workflow. The Situation Interpreter (SI) is the LangGraph graph, with one instance (and one checkpointed thread) per situation. The system further consists of one controller, which is plain deterministic code (it does not use LangGraph), and multiple agents that operate only upon request. The agents communicate via a pre-defined signal schema defined as Pydantic models. A response of an agent that does not validate against its schema is treated like a response that did not arrive (see the timeout handling below).

### Component types and conventions

Every component in this document is one of the following types, which is stated in its heading:

* **Agent** - a stateless, model-based (AI) component. It operates only upon request, answers with a structured response, and never calls another component. Model backends are configurable; for the challenge each agent may be replaced by a mock.
* **Service** - a deterministic component without a model that offers data or functionality upon request (e.g. the weather forecast fetcher, the knowledge base).
* **Deterministic component** - plain code that executes fixed rules (the controller, the communication unit).
* **Workflow** - the LangGraph-based Situation Interpreter.
* **Hardware** - physical (or, for the challenge, simulated) devices.

Signals are written as `{name attribute="value" ...}`. Names are lower-case snake_case (a signal written here as `{stop stream}` is `stop_stream` in the schema). All scores are in $[0,1]$; **a higher score always means more suspicious or more dangerous**, a higher confidence means more certain. Free text (summaries, explanations) is for display and logging only: no rule, threshold or side effect ever reads it.

The system consists of the following components:

### CCTV/Audio Processor and Filter (Service, simulated)

Multiple lightweight CCTV cameras and audio sensors record different regions of the surveilled area. Each processor stores the last 5 minutes of video/audio. This processor operates in two possible modes:
* Passive: Each processor analyses a 10-frame window of video from CCTV (~5 seconds) cameras (or a corresponding window of audio). As a primary filter, it detects unusual changes. This filter should err on the side of caution, as subsequent models will cover the more thorough analyses. Upon detection of an anomaly, it notifies the controller. It passes the triggering sequence of frames (or audio clip) and, for video, a rough bounding box of the anomalous area.
* Streaming: In addition to the actions of the "Passive" mode, the processor sends a continuous stream of video/audio to the situation interpreter, starting with a back-fill of the last minute. Receiving the {stop_stream stream_id=[stream_id]} signal returns the processor to the passive mode.

**Outgoing Signals:**

* {video_event/audio_event event_id="[id]" sensor_id="[id]" area_id="[id]" start_time="[isotime]" evidence=[media] bbox=[box]} - The processor has detected an anomalous sequence of video/audio. `bbox` is only present for video. The kind of the event (video or audio) is given by the signal name.

**Accepted signals:**

* {start_stream stream_id=[stream_id]} - The processor should switch to the streaming mode (starting with the last minute of video/audio).
* {stop_stream stream_id=[stream_id]} - The processor should stop streaming and return to the passive mode.
* {get_video/get_audio start_time="[isotime]" end_time="[isotime]"} - The processor should return the video/audio between the given timestamps. May return {out_of_scope} if the timestamps are outside the stored time range (the last 5 minutes).

[Implementation note: For the challenge, this component is a stub that returns a predetermined set of anomalous frame sequences. The streaming mode is defined in the interface, but only stubbed.]


### Person Identifier (Agent)

The person identifier is identifying a person in a given image or video. It compares the person to the reference images of the known persons in the knowledge base and returns one of three verdicts: *known*, *unknown* or *not decidable*. The agent applies the pre-configured confidence threshold itself: a match below the threshold is reported as *not decidable*.

**Incoming Signals:**

* {identify_person video="[video]" object_id="[id]" bbox=[box]} - The person identifier should identify the person in the given video.
* {identify_person image="[image]" object_id="[id]" bbox=[box]} - The person identifier should identify the person in the given image.

**Outgoing Signals:**
* {identification verdict="known" person="[person_id]" confidence=[c]} - The person identifier has matched the person to a known person. The confidence $c \in [0,1]$ expresses how certain the match is and is at or above the pre-configured threshold. The response only carries the person's ID; name, familiarity, suspicion and roles are looked up in the knowledge base and can therefore not be produced by the model.
* {identification verdict="unknown"} - The person identifier detected a person in the given video/image, but it did not match any known person.
* {identification verdict="not_decidable" cause=[cause]} - The person identifier could not decide whether the person is known (e.g. because of a hood, umbrella, or similar obstructions, or because the best match is below the confidence threshold). Cause may be "clothing", "object on person", "external object", "low confidence", "unknown". If the person identifier does not respond, the SI uses the cause "unavailable".


### Object Detector (Agent)

This agent is responsible for detecting and classifying objects in the video. It should be able to detect multiple objects at once and their possible relations (A man leaving a car, a woman carrying a crowbar, etc). For people, it should also be able to determine the role (e.g. job) of a person. It classifies the anomaly as person, animal, weather/environment, *sensor artefact* or *unclear*. It may classify an anomaly as a *sensor artefact* (e.g. lens flare, compression glitch, insects on the lens) if it does not show any object. It also reports whether the view of the camera is *obscured*.

**Incoming Signals:**

* {detect_objects evidence=[media] area_id="[id]" bbox_hint=[box]} - The object detector should detect and classify the objects in the given video/image.

**Outgoing Signals:**

* {objects_detected anomaly_class=[class] objects=[objects] relations=[relations] sensor_artefact=[bool] obscured=[bool]} - The detected objects with kind, label, confidence, bounding box and, for persons, the predicted role and role confidence.

### Knowledge Base (Service, Graph Database)

A (graph-based) knowledge base should be used to store relevant information:

1. People that the system may recognize, storing name. Each person is represented by one or more images (e.g. uploaded photos or captured by the CCTV cameras). Each person should also be annotated with a familiarity and suspicion score as well as their respective roles (family members, jobs, etc.).
2. A record of all areas of the surveilled area as well of their interconnections.
3. Permission records tracking permission rules for areas (whitelist). These rules should have the shape of simple rules. Natural language examples:
  * Family members are allowed to enter all areas.
  * Delivery people may enter the entryway, but no other areas. However, a dedicated drop-off point may be configured. In that case, one or more paths from the entrance to the drop-off point are allowed.
  * People that are not familiar and have not been invited in/accompanied by a family member in the immediate past are allowed to enter the entryway for a limited amount of time (e.g. to ring the bell) but no other areas. (An invitation only lifts this time limit for the entryway; any wider access still requires another rule.)
4. A record of all sensors and their spatial position and orientation
5. A record of all people currently in the surveilled area.

**Note:** The storage and processing of personal data and image data in particular has legal implications under the GDPR, in particular, if the camera's field of view includes public areas. I will not implement this aspect as it seems outside the scope for this challenge, this limitation should be noted and considered in the design of such a system.

### Noise Interpreter (Agent)

This agent takes audio data and classifies it as human activity, animal, weather (e.g. wind, rain), technical noise (e.g. a sensor artefact or an electrical hum) or unknown.

**Incoming Signals:**

* {interpret_noise evidence=[audio] area_id="[id]"} - The noise interpreter should classify the audio.

**Outgoing Signals:**

* {noise_interpreted category=[category] confidence=[c] is_sensor_artefact=[bool]} - The audio class, with the confidence of the classification.

### Behavioural Interpreter (Agent)

The behavioural interpreter is an agent that analyses a sequence of frames showing one or more people and tries to predict their intentions and interrelations. It should be able to predict the following intents: 
* Delivery
* Pickup
* Ring the bell
* Move to area
* Invite person
* Accompany person
* Health emergency
* Unknown activity
* Suspicious activity

It receives the identity verdict and the roles (knowledge base and predicted) of each person as context and returns, for each person, ranked intents with confidences and the degree $d \in [0,1]$ of mismatch between the predicted intent and the person's role.

**Incoming Signals:**

* {interpret_behaviour evidence=[frames] persons=[person_context] area_id="[id]"} - The behavioural interpreter should analyse the behaviour of the given persons.

**Outgoing Signals:**

* {behaviour_interpreted persons=[behaviours] relations=[relations] health_emergency=[bool]} - The ranked intents and role/intent mismatch per person, the relations between persons (accompanies, invites, ...) and whether a health emergency is suspected.

### Weather Forecast Fetcher (Service)

This service fetches the local weather forecast from a pre-configured weather forecast service. To limit the traffic to outside systems, it requests an update at most once an hour and otherwise returns cached data.

**Incoming Signals:**

* {get_forecast at="[isotime]"} - Return the forecast valid at the given time.

**Outgoing Signals:**

* {forecast entries=[entries] fetched_at="[isotime]" from_cache=[bool]} - The forecast entries (conditions and wind speed per validity interval).

### Weather Interpreter (Agent)

This agent takes video and audio data and analyzes the weather conditions and annotates them with the following information:
* High wind (with approximate wind speed)
* Snowfall
* Fog
* Rain
* Cloudy
* Sunny
* Unknown weather conditions

**Incoming Signals:**

* {interpret_weather evidence=[media] area_id="[id]"} - The weather interpreter should analyse the weather conditions.

**Outgoing Signals:**

* {weather_interpreted conditions=[conditions]} - The observed conditions with confidence (and wind speed for high wind).

### Speaker (Hardware)

Speakers are placed throughout the surveilled area.

**Incoming Signals:**

* {speak text="[text]"} - The speaker should speak the given text.

### Controller (Deterministic component)

This component is a central controller. The controller merely has an executive function. It is plain deterministic code: it follows the described workflow, is not a graph, and only ever *triggers* an LLM-based component (an agent) when indicated. It catches the {video_event/audio_event} signals of all sensors, assigns a unique situation ID, and either initializes a new Situation Interpreter for it or, if a situation for the same region is already running, forwards it to that one as an {event} signal. It receives the {situation_summary} signal from possibly multiple Situation Interpreters and amalgamates them into a single situation summary. It logs the individual situation summaries and the aggregated situation summary as well as the corresponding data.  Whenever the controller triggers a warning or an alarm, the situation and all relevant data is stored in the event log.

An {alarm} signal is triggered if the aggregated situation summary indicates a high suspicion level. Alarms are only ever triggered by the controller through the deterministic rules described here, by a user elevating a warning, or by the fallback policy for an unanswered orange warning (see below); no agent output, in particular no free text, can trigger an alarm directly.

The controller collates all information from all active Situation interpreters, calculates a combined suspicion score for each detected person. If a person exceeds a pre-configured warning threshold of suspicion (suggested default: 0.4), the controller triggers a {warning cause="suspicious person"} signal. If a person exceeds another, higher pre-configured threshold of suspicion (suggested default: 0.85), the controller triggers an {alarm cause="strong suspicion"} signal instead. These two thresholds are independent of the colour thresholds of the communication unit below. The Controller also compares the area in which a person is detected with the areas in which the person is allowed to enter. If the person is not allowed to enter the area in question *and* their suspicion has not already reached the alarm threshold above, the controller triggers a {warning cause="unpermitted entry"} signal to the log and a {speak text="You are entering without permission. Please leave the area immediately."} signal is sent; if the entry is prolonged, it will issue an {alarm cause="unpermitted entry"} signal. (A person already alarmed for strong suspicion is not also given this warning: a warning asks for a human decision that an already-triggered alarm has made moot.) Every warning carries the suspicion level of the situation summary it originates from.

Finally, the controller stores the data of all detected people in the knowledge base, alongside their roles and suspicion scores, unless the person was marked as unidentifiable (*not decidable*, except for the cause "low confidence").

If a high-danger animal (e.g. a bear) is detected by some SI, an {alarm cause="dangerous animal"} should be triggered under any of the following conditions:
  * The animal's own danger score is already at the top of the scale (0.95, e.g. a bear) -- too dangerous to leave as a warning regardless of the context below.
  * There is an open entry point to the house.
  * (Optional; default: False) One or more people are within the surveilled area.
  * (Optional; default: False) The event occurs within a specific time frame.

If none of these conditions is met, a {warning cause="dangerous animal"} is sent instead.

If an SI sends the {obscured} signal, the controller queries the weather forecast fetcher for the weather forecast. The results are then evaluated to determine whether the obstruction is plausible (e.g. by heavy fog, snow, rain or an eclipse). It also passes the video data to the behavioural interpreter to check whether the obstruction is plausibly caused by human activity of a person flagged as *unsuspicious* or within the scope of their role (e.g. the gardener blocking the view of a camera while trimming the hedges). If the obstruction is not plausible and the obstruction is prolonged, the controller triggers a {warning cause="vision obstructed"} signal. The controller then also queries the respective audio processor for the audio data and passes the results to the noise interpreter. If the interpretation indicates human activity and no family member is in the area in question, the controller triggers an {alarm cause="obstruction by human activity"} signal.

If an SI reports an *unclear situation* (see below), the controller treats its default score like any other suspicion score: at or above the warning threshold it triggers a {warning cause="unclear situation"} signal.

Each situation and each event is given a unique ID. Alarms, warnings and speaker signals are idempotent: each of them is identified by an idempotency key derived from the situation ID and its cause (e.g. `[situation_id]:[cause]`). Before sending one, the controller checks in a persistent store whether the key has already been used and, if so, does nothing and returns the earlier result. Otherwise, it sends the signal and stores the key. This ensures that retries, a resume from a checkpoint or several SIs reaching the same conclusion do not trigger the same signal twice. Additionally, the system checks, whether a sufficiently similar alarm (same cause, same area) has already been triggered in the immediate past, in order to avoid flooding for distinct situations.

**Signals received by the controller:** {video_event/audio_event} (from the processors), {situation_summary}, {obscured} and {situation_resolved} (from the SIs), {elevate warning_id=[warning_id]} and {dismiss warning_id=[warning_id]} (from the user).

**Signals sent by the controller:** {event} (to an SI), {warning}, {alarm} (to the communication unit and the log), {speak} (to the speakers) and the requests to the agents named above.

### Situation Interpreter (Workflow)

The situation interpreter is controlled by an internal state machine. By default, it is in an `Idle` state, that is only interrupted by an {event} signal from the Controller. It then switches to the `Interpret` state and analyses the incoming signals. Any subsequent {event} triggers regarding the same region are deferred. During the interpretation, the SI performs the following actions:

The SI first passes the information to the object detector. Based on the result, it selects the further sources dynamically and fans out in parallel to several agents depending on the kind of detected object/event:

* If one or more people are detected. 
  1. The SI queries the `person identifier` for the person's data. If a detected person is known to the system, the SI annotates the bounding boxes of each detected person with the person's name and the person's familiarity, suspicion score and role. The SI then also compares the person's predicted role with the ones in the knowledge base. A notable difference between the annotated roles and predicted ones is considered *suspicious*, wherein the difference in roles influences the intensity of that suspicion. If the `person identifier` is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the person as *not decidable* (cause "unavailable"). A *not decidable* person is not considered *suspicious* by itself, but is treated like a person that is not familiar in the permission checks.
  2. The SI then passes the image and video data to the `behavioural interpreter`. The behavioural interpreter then analyses the data and predicts the person's intentions and interrelations. It compares the predicted intentions with the predicted roles. A mismatch between the predicted intentions and predicted roles is considered *suspicious*, wherein the degree of mismatch ($d \in [0,1]$) between both characteristics influence the intensity of that suspicion. The behavioural interpreter may also detect possible health emergencies, which are also annotated. If the behavioural interpreter is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the behaviour as *unknown*.

* If an animal is detected, the SI annotates the potential dangerousness with a pre-defined danger score depending on its kind (1 for a bear, 0.8 for a boar, 0.6 for an unknown large dog, 0.4 for an unknown small dog, etc.). If the animal is not known to the system, it is annotated as *unknown*, which is by default not considered *dangerous*.

* If a weather event has been detected, the SI queries the weather forecast fetcher for the weather forecast and compares the results. If there is a significant mismatch between these results, the situation is considered suspicious and the weather event annotated accordingly. If either the weather forecast fetcher or the weather interpreter is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the weather event as *unknown*, which is by default not considered *suspicious*.

* If the anomaly was classified as a *sensor artefact* (by the object detector or, for audio, the noise interpreter), the SI annotates it accordingly. It has a threat score of 0 and is only logged.

* If the anomaly cannot be classified, i.e. the object detector classifies it as *unclear* or is not available, does not respond within a given time frame or returns an invalid response, the SI annotates the situation as *unclear situation* with a pre-configured default threat score (suggested default: 0.5). An unclear situation is never ignored silently, but it is also not an alarm by itself: it reaches the controller like any other score.

* If the vision of the CCTV systems is obscured (as reported by the object detector), the SI returns an {obscured} signal. 

The SI merges the respective responses and compiles them into a `situation summary`. It sends a {situation_summary} signal to the controller, which contains a summary of the situation and a separate threat assessment assigned with a score $[0,1]$. The score is the *maximum* of the scores of the individual objects/events, as the most threatening object determines the threat of the situation. The SI then moves to the `Observe` state, in which it reevaluates the situation at fixed intervals and on subsequent {event} signals by re-entering the `Interpret` state. Once the re-evaluation determines that the situation is resolved, it sends a {situation_resolved} signal to the controller and moves to the `Idle` state.

The state of each situation is checkpointed by the workflow after each step, keyed by the situation ID (the thread ID). After a crash or restart, the SI resumes from the last checkpoint with the same situation ID. Steps that are repeated after a resume only re-run analyses; side effects are protected by the idempotency keys described above.

### Communication Unit (Deterministic component)

Upon receiving a {warning cause="[cause]"} from the controller, the communication unit sends a text notification to all registered primary contacts and also to all contacts within the surveilled area. A warning with a suspicion level above 0.75 is displayed as an orange warning and those at or below as a yellow warning. During the night, this threshold is lowered to 0.5, as users are less likely to respond at night and unanswered orange warnings are elevated. A user then has the option to elevate the warning to an alarm, by clicking a button in the linked app, which sends an {elevate warning_id=[warning_id]} signal, or to dismiss it, which sends a {dismiss warning_id=[warning_id]} signal. Both are received by the controller. While waiting for the answer, the situation is paused (an interrupt of the workflow) and resumed under the same situation ID once an answer or the timeout arrives. If a warning is not answered within a pre-defined time frame, or if none of the recipients could be reached, the controller applies a fixed fallback policy: it dismisses yellow warnings and elevates orange warnings automatically (the resulting alarm is triggered by the controller and marked as caused by the fallback policy).

Each kind of alarm and warning has a pre-defined template for outside communication, which is completed with information from the annotated data in the signal. No messages for outside communication are autogenerated. Alarms are sent to the same recipients as warnings.

## Interface

The system provides a web interface for users to interact with the alarm system. Users can access the current output of all sensors, the log of all past events, and all past situations as well as their summary and related data. Visual data should be combined with the annotations and shown next to the corresponding bounding boxes.

The interface consists of several pages:
* The home page shows the current video streams of each camera. The audio of each can be streamed on-demand.
* The "history" page shows all past situations.
  * Each situation can be unfolded to show the stored footage of the surveilled area, with annotated bounding boxes and the corresponding situation summary.
* A "Simulation" page that allows the user to simulate one of the pre-defined scenarios. In this case, the backend should simulate all signals as if the system was runnging. Starting a new scenario resolves whatever is currently playing and takes over immediately, rather than being blocked by it.
* A "Person" page that allows the user to view all people currently in the database and their roles and pictures. NO edit functionality is needed.

Upon receiving a warning or an alarm, the user should be notified and the view should switch to a page showing the situation. The user can then choose to dismiss the warning or to elevate it to an alarm.

## Scope

These aspects are not within scope for this challenge:
* Any authentication aspects
* Any creation, editing or deletion of people, roles, areaa, permission rules,