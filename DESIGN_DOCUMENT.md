# Design

This document describes the design of a prototypical AI-driven security system.

## Components and Agents

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
* {person_identified person="[person_data]"} - The person identificator has identified a person in the given video/image.
* {person_not_known} - The person identificator could not detect a person in the given video/image, but it did not match any known person.
* {unidentifyable cause=[cause]} - The person identificator could not identify a person in the given video/image (e.g. because of a hood, umbrella, or similar obstructions). Cause may be "clothing", "object on person", "external object", "unknown".


### An object detector

This agent is responsible for detecting and classifying objects in the video. It should be able to detect multiple objects at once and their possible relations (A man leaving a car, a woman carrying a crowbar, etc). For people, it should also be able to determine the role (e.g. job) of a person.

### Knowledge Base

A (graph-based) knowledge base should be used to store relevant information:

1. People that the system may recognize, storing name. Each person is represented by one or more images (e.g. uploaded photos or captured by the CCTV cameras). Each person should also be annotated with a familiarity and suspicion score as well as their respective roles (family members, roles according to the [ESCO](https://ec.europa.eu/esco/lod/static/model.html).
2. A record of all areas of the surveiled area as well of their interconnections.
3. Permission records tracking permission rules for areas (whitelist). These rules should have the shape of simple Prolog-like rules. Natural language examples:
  * Family members are allowed to enter all areas.
  * [Messengers](https://ec.europa.eu/esco/lod/static/model.html#Messenger) may enter the entryway, no other areas. A dedicated drop-off point may be configured. In that case, one or more paths from the entrance to the drop-off point are allowed.
  * People that are not familiar and have not been invited in/accompanied by a family member in the immediate past are allowed to enter the entryway for a limited amount of time (e.g. to ring the bell) but no other areas.
4. A record of all sensors and their spacial position and orientation
5. A record of all people currently in the surveiled area.

**Note:** The storage and processing of personal data and image data in particular has legal implications under the GDPR, in particular, if the camera's field of view includes public areas. I will not implement this aspect as it seems outside the scope for this challenge, this limitation should be noted and considered in the design of such a system.

### Noise interpreter

### Behavioural interpreter

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

### Weather Forecast Fetch

This agent fetches the local weather forecast from a pre-configured weather forecast service.

### Weather interpreter

This agent takes video and audio data and analyzes the weather conditions and annotates them with the following information:
* High wind (with approximate wind speed)
* Snowfall
* Fog
* Rain
* Cloudy
* Sunny
* Unknown weather conditions

### Speaker

Speakers are placed throughout the surveiled area.

**Incoming Signals:**
    {speak text="[text]"} - The speaker should speak the given text.

### Situation Interpreter

This agent is a central controller. It operates in multiple modes:

* Idle: The controller does nothing. The {event} signal from a sensor transitions this system into the Observe state.
* Observe: Upon receiving an {event} signal, the situation interpreter (SI) analyzes the data attached video and hands it to the object detector. For each relevant object, it determines the spacial position (Todo: How) and queries the rele The following behaviour depends on the outcome of the object detector:

1. If one or more people are detected. The SI queries the person identificator for the person's data.

If a detected person is known to the system, the SI annotates the bounding boxes of each detected person with the person's name and the person's familiarity, suspicion score and role. The SI then also compares the person's predicted role with the ones in the knowledge base. A notable difference between the annotated roles and predicted ones is considered *suspicious*, wherein the difference in roles influences the intensity of that suspicion.

The SI then passes the image and video data to the behavioural interpreter. The behavioural interpreter then analyses the data and predicts the person's intentions and interrelations. It compares the predicted intentions with the predicted roles. A mismatch between the predicted intentions and predicted roles is considered *suspicious*, wherein the degree of mismatch ($d \in [0,1]$) between both characteristics influence the intensity of that suspicion.

The SI also compares the area in which a person is detected with the areas in which the person is allowed to enter. If the person is not allowed to enter the area in question, the SI triggers a {warning cause="Unpermitted entry"} signal to the log and a {speak your are entering without permission. Please leave the area immediately."}. If the entry is prolonged, it will issue a {alarm cause="Unpermitted entry"} signal.

If a person exceeds a pre-configured warning threshold of suspicion, the SI triggers a {warning cause="Suspicious person"} signal. If a person exceeds another threshold of suspicion, the SI triggers a {alarm cause="strong suspicion"} signal.

Finally, the SI stores the data of all detected people in the knowledge base, alongside their roles and suspicion scores, unless the person was marked as unidentifiable. Suspicion scores deteriorate over time with a configurable decay rate.

2. If an animal is detected, the SI evaluates the potential danger an animal poses. If a dangerous animal (e.g. a bear) is detected, and an alarm should be triggered under each of the following conditions:
  * There is an open entry point to the house.
  * (Optional; default: False) One or more people are within the surveiled area.
  * (Optional; default: False) The event occurs within a specific time frame.

The SI also passes the image and video data to the weather interpreter and queries the weather forcast fetcher. Both results are then compared. If a weather event has been detected, the SI queries the weather forecast fetcher for the weather forecast and compares the results.

If the vision of the CCTV systems is obscured, the SI queries the weather forecast fetcher for the weather forecast. The results are then evaluated to determine whether the obstruction is plausible (e.g. by heavy fog, snow, rain or an eclipse). 
It also passes the video data to the behavioural interpreter to check whether the obstruction is plausably caused by human activity of an person flagged as *unsuspicious* or within the scope of their role (e.g. the gardener blocking the view of a camera while trimming the hedges).
If the obstruction is not plausible and the obstruction is prolonged, the SI triggers an {warning cause="vision obstructed"} signal. The SI then also queries the respective audio recorder for the audio data and passes the results to the noise interpreter. If the interpretation indicates human activity and no family member is in the area in question, the SI triggers an {alarm cause="obstruction by human activity"} signal.

Whenever the SI triggers a warning or an alarm, the situation and all relevant data is stored in the event log.

### Communication Unit

Upon receiving a {warning cause="[cause]"} the communication sends a text notification to all registered primary contacts and also to all contacts within the surveiled area. These users then have the option to elevate the warning to an alarm, by clicking a button in the linked app, which sends an {elevate warning_id=[warning_id]}.

