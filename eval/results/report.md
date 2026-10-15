# Evaluation report

45 questions over 4 PDFs (3 synthetic multimodal documents + the project synopsis). Evidence = correct document + page (+ content type where the fact lives only in a table/figure).

## Retrieval - text embedding `sentence-transformers/all-MiniLM-L6-v2`

| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | text H@3 | table H@3 | image H@3 | latency ms |
|---|---|---|---|---|---|---|---|---|
| dense | 0.667 | 0.857 | 0.881 | 0.768 | 0.733 | 0.846 | 1.000 | 25 |
| dense+bm25 | 0.690 | 0.952 | 0.952 | 0.814 | 0.867 | 1.000 | 1.000 | 11 |
| dense+bm25+clip | 0.476 | 0.810 | 0.952 | 0.650 | 0.800 | 0.615 | 1.000 | 25 |
| hybrid+rerank | 0.905 | 1.000 | 1.000 | 0.948 | 1.000 | 1.000 | 1.000 | 554 |
| hybrid+rerank+prior0.3 | 0.857 | 0.952 | 0.952 | 0.905 | 0.867 | 1.000 | 1.000 | 551 |
| full (+modality routing) | 0.929 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 550 |
| full, reranker=ms-marco-MiniLM-L-6-v2 | 0.881 | 0.976 | 1.000 | 0.931 | 0.933 | 1.000 | 1.000 | 191 |
| full, reranker=bge-reranker-base | 0.929 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 548 |

## Retrieval - text embedding `BAAI/bge-small-en-v1.5`

| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 | text H@3 | table H@3 | image H@3 | latency ms |
|---|---|---|---|---|---|---|---|---|
| dense | 0.690 | 0.881 | 0.905 | 0.792 | 0.733 | 0.923 | 1.000 | 17 |
| dense+bm25 | 0.714 | 0.952 | 0.952 | 0.817 | 0.867 | 1.000 | 1.000 | 17 |
| dense+bm25+clip | 0.548 | 0.833 | 0.952 | 0.688 | 0.800 | 0.692 | 1.000 | 33 |
| hybrid+rerank | 0.905 | 1.000 | 1.000 | 0.948 | 1.000 | 1.000 | 1.000 | 582 |
| hybrid+rerank+prior0.3 | 0.833 | 0.952 | 0.952 | 0.893 | 0.867 | 1.000 | 1.000 | 601 |
| full (+modality routing) | 0.929 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 709 |
| full, reranker=ms-marco-MiniLM-L-6-v2 | 0.881 | 0.976 | 1.000 | 0.931 | 0.933 | 1.000 | 1.000 | 186 |
| full, reranker=bge-reranker-base | 0.929 | 1.000 | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 603 |

## End-to-end - extractive (no LLM)

| metric | value |
|---|---|
| answer accuracy (answerable) | 0.718 |
| - image questions | 0.545 |
| - table questions | 0.692 |
| - text questions | 0.867 |
| citation accuracy | 0.872 |
| abstention accuracy (unanswerable) | 0.000 |
| latency p50 / p95 (s) | 0.76 / 1.22 |

<details><summary>answers</summary>

- ✅ **Who is the CEO of Aurora Solar?** — Letter from the Chief Executive
Aurora Solar ASA is headquartered in Bergen, Norway, and employed 4,850 people at the end of the year. [1] Our Chief Executive Officer, Priya Raghunathan, joined the company in 2021 and has led its expansion from a regional installer into a pan-European solar develope
- ✅ **Where is Aurora Solar headquartered?** — Letter from the Chief Executive
Aurora Solar ASA is headquartered in Bergen, Norway, and employed 4,850 people at the end of the year. [1]
- ✅ **How many employees did Aurora Solar have at the end of 2025?** — Letter from the Chief Executive
Aurora Solar ASA is headquartered in Bergen, Norway, and employed 4,850 people at the end of the year. [1]
- ✅ **What was Aurora Solar's revenue in Q3 2025?** — |Quarter|Revenue (USD M)|Net margin|New installations (MW)| |Q3 2025|231.9|12.2%|502| [1]
- ✅ **What net margin did Aurora report for Q4 2025?** — |Quarter|Revenue (USD M)|Net margin|New installations (MW)| |Q4 2025|248.3|12.8%|540| [1]
- ❌ **How many MW of new installations were added in Q2 2025?** — 620MW [2] 237MW [2]
- ❌ **Which region had the highest installed capacity in 2025?** — Text in figure: Installed capacity by region, 2025 (MW) [1] Figure 1: Installed capacity by region (MW), 2025
4. [2]
- ❌ **What installed capacity did Iberia reach in 2025?** — 620MW [1] 237MW [1]
- ✅ **What share of revenue came from the residential segment?** — Residential 46% [1] Commercial [1]
- ❌ **What was the average module efficiency in 2025?** — Technology
Module efficiency is a key driver of the levelised cost of energy. [1] Figure 3: Average deployed module efficiency, 2019-2025 [2]
- ✅ **What cumulative capacity target has the board set for 2026?** — Outlook
For 2026, the board has set a target of 3.1 GW of cumulative installed capacity and plans to enter the Polish market in the second half of the year. [1]
- ❌ **Which new market does Aurora plan to enter?** — |Quarter|Revenue (USD M)|Net margin|New installations (MW)| |Q1 2025|182.4|9.1%|410| [3]
- ✅ **Which firmware version ships with the Helix R7?** — Introduction
The Helix R7 is a six-axis collaborative robot arm designed for machine tending, screw driving and light assembly. [1] Units ship with controller firmware HX-OS 4.2.1. [1]
- ✅ **How often must the emergency stop circuit be tested?** — Safety
The emergency stop circuit must be functionally tested every 250 operating hours and after any collision. [1]
- ✅ **What is the maximum payload of the Helix R7?** — |Parameter|Value| |Maximum payload|7 kg| [1] Helix R7 joint layout and maximum joint speeds
Joint speeds measured at rated payload
Figure 2: Joint layout of the Helix R7 [2]
- ✅ **What is the pose repeatability of the arm?** — |Parameter|Value| |Pose repeatability|±0.02 mm| [1]
- ❌ **What IP rating does the Helix R7 have?** — Introduction
The Helix R7 is a six-axis collaborative robot arm designed for machine tending, screw driving and light assembly. [2]
- ❌ **Which step comes right after the self-test in the controller startup sequence?** — Startup
When the controller boots it executes a fixed sequence of steps before any program can run. [2]
- ✅ **What is the maximum speed of joint J3?** — Text in figure: J1 Base 120°/s J2 Shoulder 120°/s J3 Elbow 180°/s J4 Wrist roll 250°/s J5 Wrist pitch 250°/s J6 Flange 320°/s [1]
- ✅ **Which joint is the shoulder?** — Text in figure: J1 Base 120°/s J2 Shoulder 120°/s J3 Elbow 180°/s J4 Wrist roll 250°/s J5 Wrist pitch 250°/s J6 Flange 320°/s [1]
- ✅ **What does error code E204 mean?** — |Error code|Meaning|Corrective action| |E204|Encoder fault|Recalibrate the joint encoder| [1]
- ✅ **How do I fix error E305?** — |Error code|Meaning|Corrective action| |E305|Communication timeout|Check the EtherCAT cable| [1]
- ✅ **How many monitoring stations were sampled on the Tamsin River?** — Background
Samples were collected from 14 monitoring stations along the Tamsin River between March and October 2024, giving 1,176 laboratory analyses in total. [1]
- ✅ **Which station had the highest mean nitrate concentration?** — Results
Nitrate concentrations varied substantially between upstream and downstream stations. [1] S1 S2 S3 S4 S5 S6 [2]
- ✅ **What nitrate limit is marked on the nitrate chart?** — Limit10mg/L [1] mg/L 6 6.8 5.3 [1]
- ✅ **What was the mean turbidity measured in the study?** — |Parameter|Mean value|Unit| |Turbidity|4.6|NTU| [1]
- ✅ **What was the mean dissolved oxygen level?** — |Parameter|Mean value|Unit| |Dissolved oxygen|7.9|mg/L| [1]
- ✅ **Which treatment plant does the study recommend upgrading?** — The study recommends upgrading the Millbrook treatment plant with tertiary nutrient removal by 2027. [1]
- ✅ **Which vector database does the multimodal RAG system use?** — This project, Multimodal RAG based AI System, focuses on designing and implementing such a system using tools that are widely adopted in the current Generative AI industry and available free of cost — LangChain for pipeline orchestration, Hugging Face Sentence-Transformers and CLIP for embeddings, P
- ✅ **Which text embedding model is specified in the technology stack?** — |11 Technology Stack|Col2| | **Text Embedding Model**|Hugging Face Sentence-Transformers — all-MiniLM-L6-v2 (free, open-source, widely used)| [1] |11 Technology Stack|Col2| |**Image Embedding Model**|CLIP — via Hugging Face Transformers (free, open-source vision- language embedding model)| [1]
- ❌ **What backend framework is used?** — 6.7 User Interface Module
A Streamlit-based web interface — one of the most widely used frameworks for building AI application front-ends — allows the user to upload documents, ask questions in natural language, and view the generated answer alongside its cited sources in a chat-like format. [3]
- ✅ **What is the minimum RAM requirement?** — ● RAM: Minimum 8 GB. [1] ● Storage: Minimum 5 GB available space. [1]
- ✅ **Over how many weeks will the project be completed?** — 15 Work Plan
The project will be completed over a 12-week schedule spanning nine phases, from requirement analysis through to final documentation. [1] Text in figure: ProjectWorkPlan-GanttChart(12-WeekSchedule) [2]
- ✅ **In the Gantt chart, which phase is scheduled for weeks 7-8?** — Phase 6:Frontend Interface&End-to-End Integration Wk 7-8 [1] The Gantt chart below shows the planned duration and sequencing of each phase by project week. [2]
- ❌ **In the system architecture diagram, which layer sits between the Backend API and the document parser?** — BackendAPI [1] The general flow of the system is as follows:
User → Uploads Document → Document Parser → Multimodal Embedding Engine → Vector Database (storage). [2]
- ✅ **Who is the faculty mentor of the project?** — Manisha Jangid Faculty Mentor [1]
- ❌ **Which modalities are out of scope for the project?** — The system is intended as an educational and demonstrative implementation rather than a production- scale enterprise deployment; nonetheless, its modular design keeps it extensible toward those directions in the future. [1] 5 Scope of the Project
The scope of the proposed project covers the design a
- ✅ **What is the title of the paper that introduced CLIP?** — “Learning Transferable Visual Models from Natural Language Supervision (CLIP).” arXiv:2103.00020. [3]
- ❌ **Which PDF parsing library is used?** — It uses a PDF-parsing library to walk through each page, extracting raw text, embedded images, and basic tabular structures, and tags each extracted item with metadata such as source filename, page number, and content type. [1] The general flow of the system is as follows:
User → Uploads Document → 
- ❌ **What dividend per share did Aurora Solar pay in 2025?** — Aurora Solar ASA - Annual Report 2025
1. [2]
- ❌ **What is the battery life of the Helix R7 teach pendant?** — Introduction
The Helix R7 is a six-axis collaborative robot arm designed for machine tending, screw driving and light assembly. [1]
- ❌ **What was the mercury concentration in the Tamsin River?** — Background
Samples were collected from 14 monitoring stations along the Tamsin River between March and October 2024, giving 1,176 laboratory analyses in total. [1]

</details>

## End-to-end - local: Qwen/Qwen2.5-1.5B-Instruct

| metric | value |
|---|---|
| answer accuracy (answerable) | 0.897 |
| - image questions | 0.727 |
| - table questions | 0.923 |
| - text questions | 1.000 |
| citation accuracy | 0.564 |
| abstention accuracy (unanswerable) | 1.000 |
| latency p50 / p95 (s) | 54.39 / 146.53 |

<details><summary>answers</summary>

- ✅ **Who is the CEO of Aurora Solar?** — The CEO of Aurora Solar is Priya Raghunathan. [1]
- ✅ **Where is Aurora Solar headquartered?** — Aurora Solar ASA is headquartered in Bergen, Norway. [1]
- ✅ **How many employees did Aurora Solar have at the end of 2025?** — At the end of 2025, Aurora Solar ASA had 4,850 employees. [1]
- ✅ **What was Aurora Solar's revenue in Q3 2025?** — According to Table 1 in the provided document, Aurora Solar's revenue in Q3 2025 was $231.9 million.
- ✅ **What net margin did Aurora report for Q4 2025?** — According to Table 1 in the provided document, Aurora reported a net margin of 12.8% for Q4 2025.
- ✅ **How many MW of new installations were added in Q2 2025?** — According to Table 1, in Q2 2025, 455 MW of new installations were added.
- ✅ **Which region had the highest installed capacity in 2025?** — According to Figure 1, Nordics had the highest installed capacity in 2025 with 620 MW.
- ❌ **What installed capacity did Iberia reach in 2025?** — According to the information provided in the sources, Iberia reached an installed capacity of 620 MW in 2025. This can be seen in Table 1 of the document, which shows the installed capacity by region for 2025.
- ✅ **What share of revenue came from the residential segment?** — According to the revenue pie chart in the document, 46% of the revenue came from the residential segment.
- ✅ **What was the average module efficiency in 2025?** — According to the information provided in the sources, the average deployed module efficiency from 2019 to 2025 was 23.6%. This can be seen in Figure 3 of the "Technology" section of the annual report, which shows the average efficiency of modules deployed in Aurora Solar ASA's projects over the spec
- ✅ **What cumulative capacity target has the board set for 2026?** — The board has set a target of 3.1 GW of cumulative installed capacity for 2026. [1]
- ✅ **Which new market does Aurora plan to enter?** — Aurora plans to enter the Polish market in the second half of the year. [2]
- ✅ **Which firmware version ships with the Helix R7?** — The Helix R7 ships with controller firmware HX-OS 4.2.1. [1]
- ✅ **How often must the emergency stop circuit be tested?** — According to the Helix R7 manual, the emergency stop circuit must be functionally tested every 250 operating hours and after any collision. [1]
- ✅ **What is the maximum payload of the Helix R7?** — The maximum payload of the Helix R7 is 7 kg [1].
- ✅ **What is the pose repeatability of the arm?** — The pose repeatability of the Helix R7 arm is ±0.02 mm. [1]
- ✅ **What IP rating does the Helix R7 have?** — The Helix R7 has an ingress protection (IP) rating of IP54. This can be found in Table 1 of the provided documentation.
- ❌ **Which step comes right after the self-test in the controller startup sequence?** — According to the controller startup sequence described in the first source, the step that comes right after the self-test is "HomeAxes LoadProgram".
- ✅ **What is the maximum speed of joint J3?** — According to the Helix R7 manual, the maximum speed of joint J3 (Elbow) is 180°/s. [1]
- ✅ **Which joint is the shoulder?** — According to the Helix R7 manual, the shoulder joint is joint J2. [1]
- ✅ **What does error code E204 mean?** — Error code E204 means an encoder fault. This can be corrected by recalibrating the joint encoder. [1]
- ✅ **How do I fix error E305?** — To fix error E305, which indicates a communication timeout, you should check the EtherCAT cable. [1]
- ✅ **How many monitoring stations were sampled on the Tamsin River?** — 14 monitoring stations were sampled on the Tamsin River. [1]
- ❌ **Which station had the highest mean nitrate concentration?** — The station with the highest mean nitrate concentration is S6, with a concentration of 12 mg/L. [1]
- ✅ **What nitrate limit is marked on the nitrate chart?** — The nitrate limit marked on the nitrate chart is 10 mg/L. This can be seen in Figure 1 of the tamsin_river_water_quality_2024.pdf document, where it shows the mean nitrate concentration by station (mg/L) with a limit of 10 mg/L.
- ✅ **What was the mean turbidity measured in the study?** — The mean turbidity measured in the study was 4.6 NTU. [1]
- ✅ **What was the mean dissolved oxygen level?** — The mean dissolved oxygen level was 7.9 mg/L. This can be found in Table 1 of the provided document [1].
- ✅ **Which treatment plant does the study recommend upgrading?** — The study recommends upgrading the Millbrook treatment plant with tertiary nutrient removal by 2027. [4]
- ✅ **Which vector database does the multimodal RAG system use?** — The multimodal RAG system uses Pinecone as the vector database. [3]
- ✅ **Which text embedding model is specified in the technology stack?** — The text embedding model specified in the technology stack is Hugging Face Sentence-Transformers — all-MiniLM-L6-v2, which is a free, open-source, widely used model.
- ❌ **What backend framework is used?** — The backend framework used is LangChain (industry-standard RAG orchestration framework).
- ✅ **What is the minimum RAM requirement?** — According to Source [1], the minimum amount of RAM required is 8 GB.
- ✅ **Over how many weeks will the project be completed?** — The project will be completed over a 12-week schedule. [1]
- ✅ **In the Gantt chart, which phase is scheduled for weeks 7-8?** — According to the Gantt chart in Figure 3, Phase 6, which involves Frontend Interface and End-to-End Integration, is scheduled for weeks 7-8.
- ✅ **In the system architecture diagram, which layer sits between the Backend API and the document parser?** — The layer that sits between the Backend API and the document parser is the Orchestration Layer. [1]
- ✅ **Who is the faculty mentor of the project?** — The faculty mentor of the project is Dr. Manisha Jangid. [1]
- ✅ **Which modalities are out of scope for the project?** — The project will not attempt to support audio or video modalities. [1]
- ✅ **What is the title of the paper that introduced CLIP?** — The paper that introduced CLIP is "Learning Transferable Visual Models from Natural Language Supervision (CLIP)" by Radford, A., Kim, J. W., et al. [4]
- ✅ **Which PDF parsing library is used?** — According to source [1], the PDF-parsing library used is PyMuPDF (fitz).
- ✅ **What dividend per share did Aurora Solar pay in 2025?** — I could not find this in the provided documents.
- ✅ **What is the battery life of the Helix R7 teach pendant?** — I could not find this in the provided documents.
- ✅ **What was the mercury concentration in the Tamsin River?** — I could not find this in the provided documents.

</details>
