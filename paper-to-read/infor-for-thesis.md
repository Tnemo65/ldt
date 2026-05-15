# Research Notes for CA-DQStream Thesis

> Auto-generated research compilation. Last updated: 2026-05-15.

---

## 1. Data Quality in Streaming (IBM, Databricks, Precisely)

### Definitions of Data Quality

- **IBM**: Data quality measures how well a dataset meets criteria for accuracy, completeness, validity, consistency, uniqueness, timeliness and fitness for purpose. It is critical to all data governance initiatives within an organization. High-quality data enables trust, which improves decision-making and leads to new business strategies. ([IBM](https://www.ibm.com/think/topics/data-quality))

- **Databricks**: Data quality measures how well data meets an organization's standards for accuracy, completeness, consistency, validity, timeliness and uniqueness. High-quality data is fit for its intended purpose, whether for analytics, AI, reporting or operational decision-making. Data integrity means data that is accurate, complete and consistent at any point in its lifecycle. ([Databricks](https://www.databricks.com/blog/what-is-data-quality))

- **Precisely**: Data quality is a vital prerequisite for any sort of analytical insight or application functionality, a significant component of data preparation, and the foundation of trustworthy data and results. Data in motion is at the most vulnerable stage—not only because of the nature of the information itself, but because of its continual fluctuation and the uncertainty of how to properly monitor the data while in transition. ([Precisely](https://www.precisely.com/blog/data-quality/big-data-quality-mastering-data-quality-in-the-age-of-big-data/))

### Key Dimensions of Data Quality

**IBM's Seven Dimensions:**
1. **Completeness** - The amount of data that is usable or complete; high percentages of missing values can lead to biased or misleading analysis
2. **Uniqueness** - The amount of duplicate data in a dataset (e.g., each customer should have a distinctive customer ID)
3. **Validity** - How much data matches the required format for business rules, including metadata, data types, ranges, and patterns
4. **Timeliness** - The readiness of data within an expected time frame (e.g., order confirmation generated in real-time)
5. **Accuracy** - The correctness of data values based on the agreed-upon "source of truth"
6. **Consistency** - Evaluating data records from two different datasets to ensure trusted insights
7. **Fitness for purpose** - Ensuring the data asset meets a business need

**Databricks' Six Dimensions:**
1. **Consistency** - Data should be consistent across different databases and datasets
2. **Accuracy** - Data should reflect the real-world scenario it's meant to represent
3. **Validity** - Data must conform to defined formats, standards, and rules
4. **Completeness** - A dataset is only as good as its completeness; missing data compromises quality
5. **Timeliness** - Data needs to be up-to-date and available when needed
6. **Uniqueness** - Account for duplications or redundancies when aggregating from various sources

### Statistics & Industry Data

- **Financial Impact**: According to Gartner, poor data quality costs organizations an average of **USD 12.9 million each year** ([IBM](https://www.ibm.com/think/topics/data-quality))
- Databricks cites similar findings: organizations lose an average of **nearly $13 million a year** as a result of poor data quality ([Databricks](https://www.databricks.com/blog/what-is-data-quality))
- **AI Dependency**: "Garbage in, garbage out" principle holds true for machine learning algorithms—if algorithms learn on bad data, they yield inaccurate results ([IBM](https://www.ibm.com/think/topics/data-quality))
- **Big Data Challenges**: The speed, variation, and enormity of big data overwhelms even organizations with rigorous data quality mechanisms ([Precisely](https://www.precisely.com/blog/data-quality/big-data-quality-mastering-data-quality-in-the-age-of-big-data/))

### Best Practices & Frameworks

**IBM Best Practices:**
- Implement data quality tools to mitigate negative impacts of poor data quality
- Conduct root cause analysis to remedy data quality issues quickly and effectively
- Designate a primary data source ("source of truth") with validation from secondary sources
- Focus on data quality management to support analytics initiatives and business intelligence dashboards

**Databricks' "Seven Cs of Data Quality" Framework:**
1. **Collect** - Capture, format, and store data in a proper repository
2. **Characterize** - Add metadata (creation time, collection method, location/sensor settings)
3. **Clean** - Address issues like duplication, typos, or unnecessary data via ETL processes
4. **Contextualize** - Determine what additional metadata may be required
5. **Categorize** - Identify key factors in datasets based on the problem domain
6. **Correlate** - Connect disparate data across various data stores
7. **Catalog** - Securely store, preserve, and make data accessible across platforms

**Quality Assessment Frameworks (Databricks):**
- Data Quality Assessment Framework (DQAF)
- Total Data Quality Management (TDQM)
- Data Quality Scorecard (DQS)
- Data downtime

**Precisely's 5-Step Framework for Data in Motion:**
1. **Discover** - Identify critical information flows, develop metric baselines, use data profiling
2. **Define** - Assess data risk, evaluate pain points, prioritize via cost-benefit analysis
3. **Design** - Create analysis and exception management processes independent of the data
4. **Deploy** - Implement controls based on criticality; includes people, processes, and technology
5. **Monitor** - Automated, continuous monitoring for data quality oversight

### Common Data Quality Challenges

- **Incomplete or inaccurate data** - Missing attributes, errors, or duplications from multiple sources
- **Poor data governance** - Unclear roles or accountability
- **Data volume and velocity** - Real-time processing challenges with growing data amounts
- **Complex data sources** - Unstructured data (photos, videos) challenges quality processes
- **Monitoring practices** - Lack of rigorous data monitoring leads to quality degradation
- **Data at rest vs. data in motion** - Companies often have processes for data at rest but fragmented solutions for data in motion

### Benefits of Good Data Quality

- **Better business decisions** - Organizations can identify KPIs and improve programs effectively
- **Improved business processes** - Identify breakdowns in operational workflows (e.g., supply chain inventory)
- **Increased customer satisfaction** - Marketing and sales teams gain insight into target buyers
- **Operational efficiency** - Reduce time spent correcting errors and addressing discrepancies
- **Enhanced data governance** - Datasets consistently managed and compliant with regulations
- **AI readiness** - High-quality data is crucial for effective AI and automation adoption

### Relevance to CA-DQStream

- **Streaming-specific alignment**: The "data in motion" vulnerability concept directly supports the need for CA-DQStream's real-time monitoring approach
- **Dimension coverage**: CA-DQStream's dimensions (completeness, timeliness, accuracy, consistency, validity) map directly to IBM and Databricks frameworks
- **Financial justification**: The $12.9M annual cost of poor data quality provides strong motivation for streaming data quality tools
- **Framework alignment**: CA-DQStream's 5W1H contextual framework aligns with Precisely's Discover-Define-Design-Deploy-Monitor lifecycle
- **Real-time requirement**: Timeliness dimension is critical for streaming systems—CA-DQStream addresses this through latency-aware scoring
- **AI dependency**: As organizations integrate AI/ML, the need for real-time data quality monitoring (as CA-DQStream provides) becomes more acute
- **Continuous monitoring**: Precisely's "Monitor" step and Databricks' emphasis on monitoring throughout the data lifecycle validate CA-DQStream's streaming approach

---

*Sources: [Signals Marketplace](https://engineering.grab.com/signals-market-place), [Data First, SLA Always](https://engineering.grab.com/data-first-sla-always), [Data Observability](https://engineering.grab.com/data-observability)*

---

### Article 1: Signals Marketplace (Data Mesh at Grab)

**What data quality challenges they address**

- High volume and variety of data across ride-hailing, food delivery, and financial services business lines
- Gaps in data ownership: no clear accountability led to ad-hoc discussions, delayed issue resolution, and teams creating duplicate pipelines to avoid relying on untrusted upstream pipelines
- Unscalable central Data Engineering bottleneck: a single DE team could not keep pace with distributed data creation/consumption
- Lack of communication between data producers and consumers: producers unaware of downstream dependencies, causing critical pipeline breaks on upstream changes
- No single source of truth: teams struggled to identify correct data definitions and reliable sources across business lines
- Varied sophistication of data practitioners across teams

**Key metrics/methods used**

- **Certification concept**: formal trust signal for high-quality, certified data assets; defines schema, SLA guarantees, freshness, and data contracts
- **Data contracts**: formal agreements between producers and consumers specifying schema, SLA guarantees (freshness, completeness, retention), notice period for changes, and communication channels
- **Data Production Incidents (DPIs)**: automated tickets created and assigned to Technical Data Owners (TDOs) when data quality tests fail on availability, timeliness, consistency, completeness, accuracy, or validity guarantees
- **Business Data Owners (BDOs)** and **Technical Data Owners (TDOs)**: clear accountability roles per data product
- **North star metric**: percentage of queries hitting certified assets (drives teams to certify most-used datasets)
- Deprecated tables increased **400% YoY** after data mesh adoption
- **75%** of Grab queries now hitting certified assets
- Number of P80 datasets (top 80% most-used data) reduced by **>58%** since certification campaign began

**System architecture components**

- **Signals Marketplace**: Grab's data mesh platform (named in 2024)
- **Data Domains**: domain-specific teams owning their data as a product
- **Genchi**: in-house data quality observational tool
- **Hubble**: metadata management platform built on DataHub with Grab proprietary technology
- Certification bootstrapped by mapping data asset creator's team to Domain ownership

**Statistics cited**

- Serving over **800 cities** across 8 Southeast Asian countries
- Deprecated tables increased **400% year-over-year** after data mesh adoption
- **75%** of Grab queries hitting certified assets
- P80 dataset count reduced by **>58%** since certification campaign start

---

### Article 2: Data First, SLA Always (Trailblazer CDC Pipeline)

**What data quality challenges they address**

- Periodic batch ETL pipelines (hourly/daily) were failing as data exceeded the petabyte threshold
- Large tables without an `updated` field made incremental ingestion impossible
- Tables without indexes on the `updated` field risked high CPU load on source databases
- Tables running full-scan strategy were silently "time bombs" — hundreds of them, potentially crashing the data system
- Spark jobs failing due to JDBC timeouts on exponentially growing tables
- Loader application was tightly coupled to upstream table characteristics; needed decoupling for true scalability

**Key metrics/methods used**

- **Change Data Capture (CDC) via MySQL Binary Logs (binlogs)**: captures all INSERT/UPDATE/DELETE events as logged events with row's past and new state
- **Debezium** running on Kafka Connect clusters to capture binlogs
- **Spark Structured Streaming** application called **Trailblazer** to persist binlogs to data lake
- **Redis cluster** for external checkpoint management: stores Kafka topic offsets as key-value pairs (key = topic name, value = JSON of partition:offset) to avoid local-disk dependency
- **DASH** (Data Auditor as a Service): hourly checks comparing ID counts between source database and streaming layer
- **maxOffsetsPerTrigger**: controls maximum messages ingested from Kafka per microbatch
- **spark.dynamicAllocation.enabled**: auto-provisions/revokes Spark executors to suit workload
- 30-second microbatch window with runtime tracking; exceeding the window indicates resource starvation
- Heartbeat emitted at end of each microbatch via StreamQueryListener to detect stale/hung jobs
- **Kafka offset divergence** monitoring: alerts when consumer lag threatens to exceed Kafka's retention window
- **Running : Active Jobs Ratio**: tracks streaming jobs registered in YARN vs actually running

**System architecture components**

- **Trailblazer**: Spark Structured Streaming CDC ingestion application
  - Binlogs captured by Debezium on Kafka Connect → Kafka cluster → Spark Structured Streaming → real-time S3 bucket → hourly/daily compaction jobs → Presto tables
- **Redis cluster**: externalized checkpoint storage (Kafka offset tracking)
- **Airflow**: orchestration for auto-retry on stream failure
- **Datadog**: metric monitoring and alerting
- **PagerDuty**: urgent on-call escalation
- **DASH**: in-house data discrepancy detection service
- **Slack**: on-call issue reporting

**Statistics cited**

- Streaming **hundreds of tables** across **60 Spark streaming jobs** at time of writing
- Data lake scaled beyond **petabyte threshold**
- Supporting **40-person Data Engineering team** providing SLA-backed data to analysts, data scientists, and ML models

---

### Article 3: Data Observability (GrabDefence Risk Systems)

**What data quality challenges they address**

- GrabDefence (in-house Risk Management platform) ingests large volumes of upstream data for real-time fraud detection and prevention
- Any data discrepancy or missing data directly impacts fraud detection and prevention capabilities
- Need for real-time (not batch) alerting on data quality issues
- Hundreds of data points make it hard to pinpoint which ones are anomalous
- JSON data from multiple sources has varying nested structures, hindering consistent analysis

**Key metrics/methods used**

- **Apache Flink SQL**: real-time stream processing and aggregation of data from multiple upstream services
- **JSONEXPLOAD** custom table function: flattens nested JSON structures into tabular rows for downstream aggregation
- **5-minute tumbling window** aggregation via Flink SQL for near real-time monitoring
- **Datadog Anomaly Detection**: out-of-the-box feature to identify unusual patterns/outliers in data streams
- **Datadog Monitor Summary**: organizes counters by service stream and underlying data points
- **Slack integration**: real-time alert notifications to Data team
- Monitoring organized by **source stream grouping** to simplify alert triage

**System architecture components**

- **Apache Flink SQL**: real-time data standardization, transformation, and aggregation
- **Datadog**: observability/monitoring platform (anomaly detection, dashboards, alerts)
- **Slack**: alert routing and team communication
- **GrabDefence**: Grab's internal risk management and fraud prevention platform

**Statistics cited**

- Data quality alerts now delivered within **same day or hour**, instead of **days to weeks** in prior system

---

### Relevance to CA-DQStream

The Grab Engineering case studies provide several directly applicable insights for CA-DQStream:

1. **Streaming data quality monitoring**: Trailblazer's DASH hourly data checks (comparing source vs streamed counts) map closely to CA-DQStream's streaming data quality evaluation — the concept of continuous, automated data discrepancy detection is a core contribution. The use of Flink SQL with 5-minute tumbling windows also mirrors the real-time aggregation windows CA-DQStream proposes.

2. **Alert organization by source/stream grouping**: Grab's grouping of hundreds of data points into service streams for simplified triage is directly applicable to how CA-DQStream might organize its quality reports — a flat list of anomalies is hard to use; grouping by source or pipeline makes it actionable.

3. **Anomaly detection as first line of defense**: Datadog's built-in anomaly detection for data stream monitoring validates CA-DQStream's core premise — automated statistical anomaly detection on data streams is an operational necessity at scale, not a luxury.

4. **Data certification and contracts**: The data contract concept (schema + SLA guarantees + freshness + completeness) in Grab's Signals Marketplace provides a framework for CA-DQStream's evaluation criteria — completeness and timeliness are explicitly mentioned as SLA dimensions tracked via automated DPI alerts.

5. **CDC as a streaming data quality baseline**: Trailblazer's architecture (binlog → Debezium → Kafka → Flink/Spark → data lake) is a reference architecture for how streaming pipelines should be instrumented; CA-DQStream's evaluation framework should assume a similar pipeline topology.

6. **Downstream impact quantification**: Grab's observation that "missing a few insert booking records in peak hours can generate wrong downstream results leading to miscalculation in revenue" provides concrete justification for the quality-relevance scoring that CA-DQStream proposes — data quality is not an academic concern but a business risk.

---

## 3b. Grab Engineering (Part 2) + Architecture

### 3b.1 Real-time Data Quality Monitoring (Coban Platform)

**Challenges Addressed:**
- **Syntactic issues:** Schema mismatches between producers and consumers causing deserialization errors
- **Semantic issues:** Field-level inconsistencies without existing enforcement frameworks (expected patterns, ranges, lengths)
- **Timeliness challenge:** No real-time mechanism to validate data against predefined rules
- **Observability challenge:** Difficulty pinpointing exact "poison data" and incompatible fields

**Architecture Components:**
- **Data Contract Definition:** Schema agreements, semantic rules, ownership details for alerting
- **Test Execution:** FlinkSQL-powered Test Runner consuming Kafka topics with dedicated consumer group
- **Result Observability:** Genchi platform for Slack notifications and Coban UI for bad record visualization

**Key Methods:**
- LLM-based semantic test rules recommendation using Kafka stream schemas and anonymized sample data
- Inverse SQL queries to capture data violating semantic rules
- Cross-field validations (planned)

**Metrics:**
- Actively monitoring 100+ critical Kafka topics
- Enables immediate identification and halting of invalid data propagation across streams

**Source:** [Real-time data quality monitoring](https://engineering.grab.com/real-time-data-quality-monitoring)

---

### 3b.2 Real-time Data Ingestion Architecture

**Challenges Addressed:**
- Data integrity issues from dual-writes to Kafka and database
- Schema maintenance burden on developers
- Burst reads on databases from SQL-based query ingestion

**Architecture Components:**
- **Stream Storage:** binlog (MySQL/Aurora) for global order; DynamoDB streams for partitioned order
- **Event Producer:** Debezium for MySQL/Aurora (Kafka Connect integration); Lambda functions for DynamoDB (auto-scaling)
- **Message Queue:** Kafka with Protobuf-encoded messages
- **Stream Processor:** Golang library consuming Kafka, writing to S3 at minute-level frequency

**Key Metrics:**
- **90% database read reduction** for Data Synchronisation Platform
- Exactly-once guarantee from stream storage

**Use Cases:**
1. **Data pipelines:** Incremental queries spanning periods vs. burst reads
2. **Drive business decisions:** Real-time data for Saga pattern microservices
3. **Database replication:** Incremental replication for disaster recovery using Strangler fig pattern
4. **Audit trails:** Regulatory compliance and fraud detection

**Source:** [Real-time data ingestion](https://engineering.grab.com/real-time-data-ingestion)

---

### 3b.3 Streaming Data Exploration (Zeppelin Integration)

**Challenges Addressed:**
- Difficulty exploring Online Data before Data Lake ingestion
- Lack of tool adoption for testing application logic on streaming data
- Schema discovery issues for downstream users

**Architecture Components:**
- **Apache Zeppelin:** Web-based notebook with Flink interpreter
- **Flink Session Cluster:** Job manager + task managers for query execution
- **DDL Derivation Tool:** Maps Protobuf schema to SQL DDL dynamically
- **Security:** Strimzi + OPA for authorization, mTLS for authentication

**Key Features:**
- SQL queries on Kafka topics without full pipeline deployment
- Schema visibility at Kafka stage (e.g., country code format detection)
- Interactive ad hoc queries becoming deployed streaming pipelines
- Audit trails for data access

**Benefits:**
- Reduces inertia in setting up development environments
- Faster reaction to upstream data producer changes
- Foundation for SQL-as-unified-streaming-language

**Source:** [Rethinking Stream Processing: Data Exploration](https://engineering.grab.com/rethinking-streaming-processing-data-exploration)

---

### 3b.4 FlinkSQL Interactive Platform

**Challenges Addressed:**
- Flink version maintenance (Zeppelin 1.17 vs. Flink 1.20+)
- 5-minute cold start delay for Zeppelin clusters
- Integration challenges with internal platforms

**Architecture (3 Layers):**
1. **Compute Layer:** Shared FlinkSQL gateway cluster (version-aligned with Flink distribution)
2. **Integration Layer:** Custom control plane with REST APIs, authentication, session management
3. **Query Layer:** Interactive UI with Hive Metastore catalog translating Kafka topics to tables

**Key Metrics:**
- Cold start reduced from **5 minutes to 1 minute**
- Full pipeline deployment in **under 10 minutes**

**Productionisation:**
- Configuration-based pipeline creation with SQL business logic
- Configurable connectors for Kafka and internal feature stores
- Dynamic JAR parsing into Flink job graphs

**Use Cases:**
- Fraud analysts debugging real-time transaction patterns
- Data scientists validating prediction models with live signals
- Engineers confirming message structure and delivery accuracy

**Source:** [The Complete Stream Processing Journey on FlinkSQL](https://engineering.grab.com/the-complete-stream-processing-journey-on-flinksql)

---

### 3b.5 AutoMQ Migration for Kafka Infrastructure

**Challenges Addressed:**
- Difficulty scaling compute resources (partition movement spikes)
- Disks couldn't scale independently (operational complexity)
- Over-provisioning based on peak usage (resource waste)
- High-risk, prolonged partition rebalancing (6+ hours latency impact)

**Architecture Components:**
- **AutoMQ:** Cloud-native Kafka with shared storage architecture
- **EBS WAL:** Fixed 10GB EBS for single-digit millisecond write latency
- **S3:** On-demand object storage for data persistence
- **Strimzi Operator:** Extended for AutoMQ WAL volume management

**Key Metrics:**
- **3x throughput increase** per CPU core
- **3x cost efficiency improvement**
- Partition reassignment: **6 hours → <1 minute**

**Technical Benefits:**
- No partition data movement during scaling (shared storage across brokers)
- On-demand S3 storage (no manual disk scaling)
- Fast reassignment via metadata synchronization only
- Eliminates inter-broker replication overhead
- Reduced I/O and network utilization spikes
- No prolonged latency increase for producers/consumers

**Future Enhancements:**
- Self-Balancing feature (similar to Cruise Control)
- Auto-scaling with spot instances
- S3 WAL mode to reduce cross-AZ traffic
- Table Topics for direct iceberg table format storage

**Source:** [How Grab Uses AutoMQ to Solve Kafka Challenges](https://www.automq.com/blog/how-grab-uses-automq-solve-kafka-challenges)

---

### Relevance to CA-DQStream

The second set of Grab Engineering case studies provides additional architectural insights for CA-DQStream:

1. **Data Contract Framework:** Grab's Coban platform demonstrates syntactic + semantic test rules, similar to data quality contracts needed in CA-DQStream for schema compliance and semantic validation.

2. **FlinkSQL for Quality Monitoring:** FlinkSQL-based Test Runner validates streaming data in real-time, directly applicable to CA-DQStream's quality assessment approach using windowed aggregations.

3. **LLM-Assisted Rule Generation:** LLM-based semantic rule recommendation (Coban) shows potential for automating quality rule creation in CA-DQStream — reducing manual effort in defining evaluation criteria.

4. **Separation of Storage and Compute:** AutoMQ's architecture demonstrates benefits of separating streaming storage from processing, which aligns with CA-DQStream's design of independent quality evaluation components.

5. **Observability Integration:** Genchi + Coban UI + Slack alerting provides model for CA-DQStream's result presentation layer — combining automated detection with human-in-the-loop response.

6. **Debezium CDC Pattern:** Real-time ingestion via binlog → Kafka provides reference architecture for streaming pipeline instrumentation that CA-DQStream should assume.

7. **Multi-layered Architecture:** FlinkSQL gateway's Compute/Integration/Query separation offers pattern for CA-DQStream's modular design separating data ingestion, quality evaluation, and presentation layers.


1. [Data Quality in Streaming](#1-data-quality-in-streaming)
2. [Streaming Architectures & Technologies](#2-streaming-architectures--technologies)
3. [Real-World Case Studies (Grab, Uber)](#3-real-world-case-studies-grab-uber)
4. [Concept Drift Detection](#4-concept-drift-detection)
5. [Anomaly Detection Methods](#5-anomaly-detection-methods)
6. [Baseline Methods for Comparison](#6-baseline-methods-for-comparison)
7. [Context-Aware Systems](#7-context-aware-systems)
8. [Citation-Ready References](#8-citation-ready-references)

---

## 3d. Grab Kafka Data Quality (InfoQ)

### Data Quality Monitoring at Grab

- Grab, a Singapore-based digital service delivery platform, added data quality monitoring to its **Coban internal platform** to improve the quality of data delivered by Apache Kafka to downstream consumers
- Deployed earlier in 2025, the system now actively monitors data quality across **100+ critical Kafka topics**
- Key achievement: "the solution offers the capability to immediately identify and halt the propagation of invalid data across multiple streams... This accelerates the process of diagnosing and resolving issues, allowing users to swiftly address production data challenges"
- Industry benchmark: Only an estimated **1% of companies** have reached the highest maturity level where "data streaming is a strategic enabler with streams managed as a product" (Confluent 2025 Data Streaming Report)

### Schema Validation Approaches

- Two types of data errors identified:
  - **Syntactic errors**: Errors in message structure (e.g., string value for a field defined in schema as int, causing deserialization errors)
  - **Semantic errors**: Data values that are poorly structured or outside acceptable limits (e.g., user_id field violates expected format of 'usr-{8-digits}')
- Core system architecture: **test configuration and transformation engine** that takes topic data schemas, metadata, and test rules as inputs
- Uses **LLM to analyze Kafka stream schemas and anonymized sample data** to recommend potential semantic test rules, dramatically accelerating setup process and identifying non-obvious data quality constraints
- **FlinkSQL-based test definitions** generated automatically from schemas and rules

### Architecture Patterns

- **Test Configuration and Transformation Engine**: Takes schemas, metadata, and test rules → creates FlinkSQL-based test definitions
- **Flink Job Execution**: Consumes messages from production Kafka topics and forwards errors to observability platform
- **FlinkSQL selected** because its ability to represent stream data as dynamic tables allowed automatic generation of data filters
- **LLM-assisted rule recommendation**: Analyzes schemas and sample data to suggest semantic rules

### Key Metrics/Statistics

| Metric | Value |
|--------|-------|
| Critical Kafka topics monitored | 100+ |
| Industry maturity (top tier streaming) | ~1% of companies |

### Additional Context: Grab's Near Real-Time Data Lake Architecture

- **Hudi format** (Merge On Read / MOR) for minimal data latency on data lake
- **Flink** for stream processing, **Spark** for compaction (Avro → Parquet conversion)
- **Protobuf** as central data format in Kafka with schema evolution compatibility
- **Partitioning by Kafka event time** up to hour level for optimized Hudi operations
- **Flink CDC connectors** (Debezium) for RDS data ingestion from MySQL binlog
- **Indexing strategies**: Bucket Index vs Flink State Index for Hudi upserts
- **Impact achieved**: Minute-level data latency, enabling fresh business metrics dashboards and quicker fraud detection

### Relevance to CA-DQStream

- **Direct alignment**: Grab's FlinkSQL-based data quality monitoring validates our approach of using stream processing engines for real-time validation
- **Two-tier error taxonomy** (syntactic/semantic) provides a useful framework for CA-DQStream's own error categorization
- **LLM-assisted rule generation** complements CA-DQStream's context-aware detection by suggesting domain-specific constraints
- **Schema evolution handling** with Protobuf aligns with our need to handle dynamic schema changes in streaming pipelines
- **100+ topic monitoring scale** demonstrates feasibility of production deployment at scale, relevant for CA-DQStream evaluation benchmarks
- **End-to-end latency reduction** (to minute level) achieved via Hudi/MOR validates our streaming architecture design for thesis experiments

---

## 2a. Confluent Streaming Data Quality

This section summarizes key patterns, architecture recommendations, and principles from Confluent's official blog posts on streaming data quality and schema management.

---

### Data Quality Patterns for Streaming

**From: [Preventing and Fixing Bad Data in Event Streams, Part 1](https://www.confluent.io/blog/shift-left-bad-data-in-event-streams-part-1/)**

- **Eight main types of bad data in event streams:**
  1. Corrupted data (garbled bytes, faulty serializers)
  2. Events with no schema (no structure to define "good" vs "bad")
  3. Events with invalid schema (Schema ID doesn't correspond to a valid schema in registry)
  4. Incompatible schema evolution (breaking changes that consumers can't handle)
  5. Logically invalid field values (e.g., array of integers for "first_name", null in NPE-declared fields)
  6. Logically valid but semantically incorrect (e.g., SQL injection strings in name fields, negative costs)
  7. Missing events (no data was produced but should have been)
  8. Duplicate/unwanted events (bug-induced duplicates that can't be distinguished from real events)

- **Prevention is the single most effective strategy** — "an ounce of prevention is worth a pound of cure"; catching bad data early prevents contamination spreading to downstream systems

- **Contamination cascades quickly**: Once bad data enters a stream, it spreads to all consumers, contaminating every dependent dataset; fixing requires surgical removal + reprocessing + downstream recomputation

- **Dead-letter queue (DLQ) caution**: Shunting data to DLQs is a last resort — it preserves an error-free but *incomplete* stream, which can cause miscalculations; best when each event is independent and ordering doesn't matter

- **Incremental processing is not immune**: Bad data inputs to incremental jobs require unclick/reverse operations; when reversibility is complex, full rebuild is safer (even dbt recommends "full_refresh" for misbehaving incremental models)

- **Three-tier mitigation hierarchy:**
  1. **Prevention** — schemas, testing, validation rules; fail fast at ingestion
  2. **Event Design** — design events that allow issuing correction events to overwrite bad data
  3. **Rewind, Rebuild, Retry** — when all else fails

**From: [Making Data Quality Scalable With Real-Time Streaming Architectures](https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/)**

- **Shift-left data quality**: Moving validation checkpoints to the source (ingestion time) rather than at downstream destinations — stops bad data before it spreads, rather than reacting after contamination

- **Two-layer continuous quality framework:**
  1. **Validation** — ensuring data conforms to expected structure and meets completeness/accuracy rules while in motion
  2. **Monitoring** — tracking ongoing health metrics to spot trends, detect anomalies, and intervene before issues reach downstream systems

- **Key quality KPIs to track continuously:**
  - Completeness: % of required fields populated
  - Accuracy: values within expected ranges
  - Freshness: event arrival vs. processing time lag
  - Error rates: invalid events vs. total volume
  - Quarantine volume: events routed for review/correction

- **Batch validation limitations**: Traditional batch ETL checks validate at set intervals (hourly/daily) — by the time errors are flagged, damage has already propagated to payment processing, financial reporting, and fraud detection models

- **Real-time validation benefits:**
  - Errors blocked immediately at ingestion
  - Reduced mean time to resolve (MTTR) for data incidents
  - Lower reprocessing workloads
  - Improved dashboard trust scores
  - Stronger compliance posture

- **Data quality feedback loop**: Validation → routing invalid records to quarantine → monitoring dashboards → threshold-based alerting → corrective action

- **Proactive alert example**: If >2% of events fail validation within a 5-minute window, engineers are notified immediately to prevent silent data errors from cascading

---

### Schema Validation Approaches

**From: [Preventing and Fixing Bad Data in Event Streams, Part 1](https://www.confluent.io/blog/shift-left-bad-data-in-event-streams-part-1/)**

- **Use explicitly-defined schemas (Avro, Protobuf, or JSON Schema)** — JSON is a common but poor choice for events because it doesn't enforce types, mandatory/optional fields, defaults, or evolution rules; going schemaless puts the burden entirely on consumers to interpret data

- **Implicit schemas, tribal knowledge, and conventions are not suitable** for providing data integrity — strict schemas reduce consumer exposure to unintentional data issues

- **Multi-consumer risk amplification**: Without schemas, every consumer-topic pair is a chance for misinterpretation; N topics × M consumers = N×M chances for divergent data interpretation, leading to silent errors (miscalculated sums, misattributed results) that are harder to detect than loud exceptions

- **Data Quality Rules via Confluent Data Contracts**: CEL-based (Common Expression Language) rules enforced at serialization time; example SSN rule:

  ```json
  {
    "name": "checkSsnLen",
    "kind": "CONDITION",
    "type": "CEL",
    "mode": "WRITE",
    "expr": "size(message.ssn) == 9"
  }
  ```

- **Rule enforcement modes**: WRITE (reject at producer) or ON_READ (allow through but flag); WRITE mode is preferred for prevention

- **Unit and integration testing for producers**: Test serializers/deserializers, schema formats (validate against production Schema Registry), data validation rules, and business logic; integrate into CI/CD pipeline

**From: [Best Practices for Kafka Connect Data Transformation & Schema Management](https://www.confluent.io/blog/kafka-connect-data-transformation-schema/)**

- **Schema format selection:**
  - Avro: compact binary format, ideal for high-throughput systems
  - Protobuf: efficient serialization, broad adoption in microservices
  - JSON Schema: human-readable and widely understood, less compact

- **Schema Registry compatibility modes:**
  - BACKWARD: new schemas can read data written with older schemas (consumers updated first)
  - FORWARD: old schemas can read data written with newer schemas (producers updated first)
  - FULL: both backward and forward compatibility — use for critical systems
  - NONE: no compatibility checks enforced

- **Schema evolution best practices:**
  - Add new fields with default values (never remove required fields)
  - Mark fields as deprecated before removing them (give consumers time to migrate)
  - Prefer adding new fields over modifying or deleting existing ones
  - Update schemas incrementally
  - Integrate schema validation into CI/CD pipeline to catch compatibility issues early

- **Single Message Transforms (SMTs) for schema-level operations:**
  - Filter messages based on field values
  - Mask PII fields (e.g., credit card numbers, email addresses)
  - Rename fields before writing to sink
  - Cast data types (e.g., string to integer)
  - Insert metadata fields (timestamps, source identifiers)

- **Consistent converter configuration**: Configure Kafka Connect to use the same converters (Avro, Protobuf, JSON Schema) as producers and consumers — misaligned converters cause serialization/deserialization errors

---

### Architecture Recommendations

**From: [Making Data Quality Scalable With Real-Time Streaming Architectures](https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/)**

- **Six-step real-time validation pipeline architecture:**
  1. **Ingest** data into Kafka topics (acts as backbone of the pipeline)
  2. **Enforce schema validation** via Schema Registry at ingestion (reject malformed events immediately)
  3. **Apply business rules** with Apache Flink or ksqlDB (out-of-range values, missing IDs, abnormal transaction spikes)
  4. **Route invalid data to quarantine** (dead-letter topic for inspection, correction, and reprocessing)
  5. **Monitor data quality KPIs** with Grafana/Looker/Datadog dashboards (freshness, error rates, completeness)
  6. **Trigger alerts** on threshold breaches (e.g., >2% failure rate in 5-minute window)

- **Continuous quality monitoring**: Treat data quality as a continuous process, not a one-time task — build validation and monitoring directly into the data's journey

- **Monitoring integration**: Export quality metrics to existing observability stacks (Grafana, Datadog) alongside system health and infrastructure metrics

**From: [Best Practices for Kafka Connect Data Transformation & Schema Management](https://www.confluent.io/blog/kafka-connect-data-transformation-schema/)**

- **Exactly-once semantics (EOS)**: Use Kafka Connect with EOS when supported by connectors — combines idempotent producers with transactional writes to ensure each record is processed exactly once even under failures

- **Two-phase commit (2PC) strategy**: Ensures atomicity — transactions are either fully committed or entirely rolled back across all involved systems (Kafka + external databases/APIs)

- **Idempotent operations design:**
  - Source connectors: ensure Kafka only consumes committed changes from external systems; use Debezium for transactional consistency
  - Sink connectors: use upsert modes with unique keys (primary keys, unique IDs) for idempotent writes

- **Transactional API leverage**: Many external systems (PostgreSQL, RabbitMQ, Amazon S3) provide transactional APIs — configure Kafka Connect to group multiple writes into a single atomic operation; offset committed only if external operation succeeds

- **Error tolerance configuration:**
  - `errors.tolerance = 'none'` (default): connector task fails immediately on any error (critical use cases)
  - `errors.tolerance = 'all'`: skip problematic messages and continue (enables DLQ usage)
  - `errors.retry.timeout`: max duration for retry attempts
  - `errors.retry.delay.max.ms`: max delay between retries (use with exponential backoff)

- **Enrichment architecture:**
  - Lightweight enrichment (timestamps, metadata, field renaming): use SMTs in Kafka Connect
  - Complex enrichment (lookups, joins, real-time processing): use Apache Flink or Kafka Streams (Connect is not designed for heavy traffic)

- **PII data handling:**
  - Masking: use custom SMTs to redact sensitive fields dynamically
  - Encryption: SSL/TLS for in-transit encryption; server-side encryption for storage sinks (S3)
  - Policy as code: embed data contracts as code for automated enforcement and audit trails

---

### Key Principles and Statistics

**From: [Making Data Quality Scalable With Real-Time Streaming Architectures](https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/)**

- **Bad data costs are real and immediate**: Show up as inaccurate dashboards, failed compliance audits, customer churn, and wasted operational resources — not just theoretical
- **Batch validation's hidden cost**: Vimeo experienced a one-day delay before insights reached analytics teams, limiting real-time decision-making and quick pivots after launches/campaigns (Babak Bashiri, Director of Data Engineering, Vimeo)
- **Enterprise scale example**: Siemens Healthineers processes 8 million messages daily via Confluent, catching manufacturing defects instantly and enabling reliable diagnostic results for patients
- **Shift from reactive to proactive**: Organizations that shift left (validate at source) see reduced MTTR, less downstream rework, and improved dashboard trust scores
- **Industry applicability**: Financial services (real-time fraud detection), retail (multi-channel pricing consistency), healthcare (patient data accuracy + HIPAA compliance), AI/ML (clean training data to prevent model drift)

**From: [Preventing and Fixing Bad Data in Event Streams, Part 1](https://www.confluent.io/blog/shift-left-bad-data-in-event-streams-part-1/)**

- **Schema adoption reduces error incident rates significantly** by preventing producers from writing bad data, freeing consumers to focus on using data rather than parsing its meaning
- **Immutability is a feature, not a bug**: Kafka's immutability means every consumer gets the same auditable data, but it requires deliberate data creation upfront
- **Silent errors are more dangerous than loud ones**: Misinterpretations of schemaless data lead to miscalculated sums and misattributed results — undetected divergence across teams (e.g., one team's engagement report doesn't match another's billing report)
- **CI/CD schema validation**: Once schemas are adopted, CI/CD pipelines can perform schema, data, and evolution validation before deployment — no more spewing bad data into production streams

---

### Relevance to CA-DQStream

- **Strongly validates the "shift-left" core principle**: CA-DQStream's context-aware validation at ingestion aligns with Confluent's recommendation to enforce quality checks at the source, not at downstream batch jobs
- **Multi-layered validation model matches CA-DQStream architecture**: Schema enforcement (structural) + business rule checks (semantic) + monitoring/alerting mirrors CA-DQStream's layered quality framework
- **Dead-letter queue pattern**: CA-DQStream's quarantine mechanism for anomalous windows aligns with Confluent's DLQ best practice for preserving completeness while isolating bad data
- **CEL-based rule expression**: Confluent's CEL rules for semantic validation (e.g., `size(message.ssn) == 9`) provide a concrete reference for CA-DQStream's rule evaluation engine
- **Contamination cascade problem**: Confluent's observation that bad data "spreads quickly and contaminates everything it touches" directly motivates CA-DQStream's window-relative anomaly detection — catching contamination early within bounded time windows
- **Schema evolution compatibility**: CA-DQStream should support BACKWARD/FORWARD/FULL compatibility modes for schema versioning across evolving stream definitions
- **DLQ caution as design insight**: The warning that DLQs create error-free but *incomplete* streams (potentially causing miscalculations) supports CA-DQStream's need for completeness scoring alongside anomaly detection
- **Key gap CA-DQStream addresses**: Confluent focuses on structural and rule-based validation; CA-DQStream's context-aware detection (learned temporal patterns, periodicity awareness, concept drift) extends this to semantic and behavioral anomaly detection that schemas and rules cannot capture.

## 2b. NYC Taxi Streaming Architecture (Kafka + Flink)

### Overview

The NYC Taxi and Limousine Commission (TLC) trip record data is a canonical benchmark dataset for streaming pipeline research, featuring millions of taxi trips with detailed fare, location, and timing information. Multiple production systems have been built to stream, process, and analyze this data using Apache Kafka and Apache Flink.

### Official Data Source

**NYC TLC Trip Record Data** ([nyc.gov](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page))

| Dataset | Description | Fields |
|---------|-------------|--------|
| Yellow taxis | Standard NYC yellow cabs | Pickup/drop-off datetime, locations, trip distance, itemized fares, rate types, payment types, passenger counts |
| Green taxis | NYC green cabs | Same schema as yellow taxis |
| For-Hire Vehicles (FHV) | Base-dispatched vehicles | Dispatching base license, pickup datetime, taxi zone location ID |
| High Volume FHV (HVFHS) | Uber/Lyft-style platforms | Detailed data since 2019, congestion fee column added 2025+ |

- **Format**: PARQUET (monthly, published with ~2 month delay)
- **Access**: Direct download or NYC Open Data Portal

### Architecture Patterns for Streaming Taxi Data

#### Kappa Architecture with Kafka + Flink

Several projects implement streaming ETL pipelines for NYC taxi data using a simplified Kappa-style architecture:

```
CSV/Parquet ──▶ Python ──▶ Apache ───▶ Apache ───▶ Elasticsearch
Historical     Producer    Kafka      Flink       + Kibana
Data           (Simulated  Topic      Consumer
               Real-time)  taxi_trips (Processing)
```

**Source**: [NYC Taxi Streaming - techatspree](https://github.com/techatspree/nyc-taxi-streaming)

#### Key Components

| Component | Technology | Role |
|----------|------------|------|
| Message Broker | Apache Kafka | Real-time event streaming, topic partitioning |
| Stream Processing | Apache Flink | Stateful stream processing, windowed aggregations |
| Message Routing | Apache Camel | Optional routing logic |
| Visualization | Elasticsearch + Kibana | Real-time dashboards |
| Containerization | Docker | Reproducible deployment |

### Python Kafka Consumer for NYC Taxi Data

**Producer Pattern** (Simulating Real-Time Taxi Events):

```python
import json
import time
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "taxi_trips"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv("data/samples/sample_1k.csv")

for _, row in df.iterrows():
    event = row.to_dict()
    producer.send(TOPIC, event)
    time.sleep(1)  # 1 event per second simulation
    producer.flush()
```

**Spark Structured Streaming Consumer**:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("VendorID", StringType()),
    StructField("tpep_pickup_datetime", StringType()),
    StructField("tpep_dropoff_datetime", StringType()),
    StructField("passenger_count", DoubleType()),
    StructField("trip_distance", DoubleType()),
    StructField("PULocationID", StringType()),
    StructField("DOLocationID", StringType()),
    StructField("fare_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
])

spark = SparkSession.builder \
    .appName("TaxiTripConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1") \
    .config("spark.sql.streaming.checkpointLocation", "/data/checkpoint") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "taxi_trips") \
    .option("startingOffsets", "earliest") \
    .load()

parsed_df = df.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")).select("data.*")

query = parsed_df.writeStream \
    .format("json") \
    .option("path", "/data/bronze") \
    .option("checkpointLocation", "/data/checkpoint") \
    .outputMode("append") \
    .trigger(processingTime="10 seconds") \
    .start()

query.awaitTermination()
```

**Source**: [Phase 1 NYC Taxi Streaming Pipeline - Hamza Paracha](https://medium.com/@hamzaparacha098/phase-1-part-2-building-a-real-time-streaming-pipeline-with-kafka-spark-python-for-nyc-63f2471dedbd)

### Flink Operator Patterns for Taxi Data Processing

#### Windowed Aggregations

```scala
val fullRides = messages.map(tr => (tr.ride_id, List(tr)))
    .reduceByKeyAndWindow(
        (first, second) => first ++ second,
        (first, second) => second,
        Seconds(30),      // window duration
        Seconds(10)       // slide interval
    )
    .filter { e =>
        e._2.map(_.point_idx).sorted.sliding(2).forall(l => l.head+1 == l.tail.head)
    }
    .map { e =>
        (e._1, 
         calculateDistance(e._2),        // Haversine formula
         calculateMeterIncrementSum(e._2),
         calculateTravelTimeInSeconds(e._2))
    }
```

#### Feature Engineering for Anomaly Detection

Key features computed per taxi ride window:

| Feature | Calculation | Use Case |
|---------|-------------|----------|
| `distance` | Haversine formula (lat/lon start → end) | Trip validity |
| `meterIncrement` | Sum of meter increments | Fare validation |
| `meterDiff` | Last - first meter reading | Anomaly detection |
| `travelTime` | Timestamp difference (seconds) | Duration anomalies |
| `observations` | Count of windowed records | Data completeness |

**Source**: [Isolation Forest NYC Taxi - SoftwareMill](https://softwaremill.com/isolation-forest-anomaly-detection-with-spark-and-nyc-taxi-data-stream/)

### Anomaly Detection on Streaming Taxi Data

#### Isolation Forest Pattern

Unsupervised anomaly detection using Isolation Forest on computed ride features:

```scala
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.iforest._

val assembler = new VectorAssembler()
    .setInputCols(Array("distance", "meterIncrement", "meterDiff", "travelTime"))
    .setOutputCol("features")

val data = assembler.transform(rawData).select(col("features"), col("rideId"))

val isolationForest = new IsolationForest()
    .setNumEstimators(100)
    .setMaxSamples(256)
    .setContamination(0.1)
    .setFeaturesCol("features")
    .setPredictionCol("predictedLabel")
    .setScoreCol("outlierScore")

val model = isolationForest.fit(data)
val predictions = model.transform(data)
```

**Performance benchmark**: NYC Tycoon Pub/Sub stream generates 2000-2500 taxi ride updates per second (~8 Mb/sec)

**Anomaly detected example**:
- 34.81 USD/km vs. mean 7.06 USD/km (4.9x higher)
- 790.29 USD/hour vs. mean 97.64 USD/hour (8.1x higher)

#### Lyft-Style Real-Time Anomaly Detection

Lyft's production anomaly detection architecture for mobility data:

| Challenge | Solution |
|----------|----------|
| Anomaly detection complexity | Minute-level streaming windows |
| Data preparation | ML-based feature engineering |
| Non-JVM language support | Flink SQL jobs for cross-team access |
| Pattern recognition | SQL-based filtering and aggregation |

**Source**: [Lyft Streaming Platform - Alibaba Cloud](https://www.alibabacloud.com/blog/lyfts-large-scale-flink-based-near-real-time-data-analytics-platform_596674)

### Data Quality Monitoring in Taxi Streaming Pipelines

#### Grab's Real-Time Quality Framework (Applicable Pattern)

```
Data Contract ──▶ Test Runner ──▶ Kafka Topic ──▶ Genchi UI +
Definition       (FlinkSQL)     (Bad Records)   Slack Alerts
    │                │
    │                └──▶ Inverse SQL queries for violations
    │
    └──▶ Schema rules, semantic rules, alert config
```

**Key metrics**:
- Sub-10ms latency per record validation
- 60-second sliding windows for quality checks
- Routes incomplete records to dead letter queues
- Preserves valid records for main pipeline

**Source**: [Grab Real-time Data Quality Monitoring](https://engineering.grab.com/real-time-data-quality-monitoring)

#### Six Quality Dimensions for Taxi Streaming

| Dimension | Definition | Taxi Example |
|-----------|------------|--------------|
| **Completeness** | Required fields present | Missing fare_amount |
| **Timeliness** | Data arrives within windows | Delayed trip reports |
| **Accuracy** | Values reflect real state | GPS coordinates off |
| **Consistency** | Format agreement across systems | Different datetime formats |
| **Uniqueness** | No duplicate records | Duplicate trip IDs |
| **Validity** | Conforms to schema/rules | fare_amount < 0 |

### Performance Benchmarks

| System | Throughput | Latency | Notes |
|--------|------------|---------|-------|
| NYC Tycoon Pub/Sub | 2000-2500 events/sec | ~8 Mb/sec | Simulated taxi data |
| Lyft Flink Platform | Minute-level | Near real-time | Production mobility data |
| Grab Quality Monitor | <10ms/record | 60-sec windows | 100+ topics monitored |
| Spark Structured Streaming | ~1000 rows/sec | 10-sec triggers | Local Python simulation |

### Key Technical Challenges and Solutions

| Challenge | Solution | Reference |
|-----------|----------|-----------|
| Simulating real-time from batch | Python producer with controlled `time.sleep()` | Paracha (2025) |
| Schema evolution | Schema Registry with backward compatibility | Confluent patterns |
| Missing trip data | Windowed aggregation with tolerance | SoftwareMill |
| Geographic anomalies | Haversine distance validation | NYC TLC zones |
| Fare validation | Statistical thresholds + Isolation Forest | SoftwareMill |
| Real-time alerting | FlinkSQL + Slack integration | Grab Coban |

### Relevance to CA-DQStream

1. **Benchmark dataset**: NYC TLC provides standardized, well-documented data for evaluating streaming quality detection — directly applicable as CA-DQStream's primary evaluation dataset.

2. **Windowed processing pattern**: The sliding window aggregations (30s window, 10s slide) used in taxi streaming directly map to CA-DQStream's temporal quality scoring.

3. **Anomaly detection integration**: Isolation Forest on computed features (distance, fare/time ratios) provides a concrete baseline method for CA-DQStream comparison.

4. **Producer simulation**: The Python-based Kafka producer pattern enables reproducible streaming experiments without live taxi infrastructure.

5. **Quality dimension mapping**: Six dimensions (completeness, timeliness, accuracy, consistency, uniqueness, validity) provide a framework for CA-DQStream's multi-dimensional scoring.

6. **Dead letter queue pattern**: Routing bad records while preserving valid streams is essential for maintaining pipeline completeness during anomaly events.

---

## 3c. Uber + Ververica/Flink Case Studies

### Uber: M3 Query Engine for High Cardinality Time Series

Source: [The Billion Data Point Challenge](https://www.uber.com/ie/en/blog/billion-data-point-challenge/)

**Scale & Performance Numbers**

- ~2,500 queries per second served
- ~8.5 billion data points returned per second
- ~3.5–35 Gbps network traffic handled (as of Nov 2018)
- Active-active multi-datacenter deployment across several cloud zones
- Scale has grown faster than Uber's organic growth due to increased metrics adoption

**Data Quality Monitoring at Scale**

- Monitoring infrastructure spans entire tech stack: from low-level system metrics (memory, CPU) to high-level business metrics (orders per city, Eats transactions)
- Tools built on top of the M3 metrics platform include anomaly detection, resource estimation, and alerting
- Batch systems like Prometheus and Graphite ceased to work at this scale, necessitating custom-built M3

**Architecture Patterns**

- M3 query engine goes through three phases: parsing → execution → data retrieval
- DAG-based query representation supporting both M3QL (tag-based, pipe-line) and PromQL natively
- Decoupled storage backend (M3DB) from execution engine for flexibility
- Columnar storage with time-block partitioning for parallel processing across blocks
- Worker pool pattern for goroutine reuse to reduce stack allocation overhead
- Lazy evaluation of query functions to minimize memory footprint — lazy column-wise evaluation uses ~161 KB vs ~1.7 GB for naive sequential decompression
- Context propagation via HTTP close notify to cancel superseded queries and prevent resource pile-up

**Anomaly Detection & Downsampling**

- LTTB (Largest Triangle Three Buckets) downsampling algorithm chosen over naive averaging to preserve outliers and peaks — critical for accurate anomaly detection
- Averaging-based downsampling was found to hide anomalies (e.g., peaks and troughs) by modifying actual data points

**Data Latency Reduction**

- Read fanout at query time (not write replication) across data centers to avoid storage cost growth proportional to datacenter count
- Returning compressed metrics directly and streaming them (vs batching) achieved 3x latency reduction for cross-datacenter queries

---

### Ververica: Apache Flink for Fintech & Stream Processing

Source: [Fintech Monitoring](https://www.ververica.com/banking/fintech-monitoring) | [What is Apache Flink](https://www.ververica.com/ecosystem-introduction/what-is-apache-flink)

**Scale & Performance Numbers**

- VERA engine sustains 6.9 billion records per second throughput
- Sub-10ms latency from event ingestion to metric computation
- Exactly-once processing guarantees — no duplicates, no gaps, even during failover and scaling
- 40% lower TCO compared to self-managed Apache Flink
- Alibaba case study: processes over 1 trillion events per day and 470+ million transactions per second at peak ([source](https://www.alibabacloud.com/blog/why-did-alibaba-choose-apache-flink-anyway_595190))
- Cost of delayed compliance detection in financial services: $4.7 billion in regulatory fines in 2025 alone

**Data Quality Monitoring at Scale**

- Real-time portfolio analytics: track valuations, risk exposures, rebalancing signals across thousands of accounts simultaneously
- Automated compliance monitoring: MiFID II suitability checks, KYC triggers, concentration limits, reporting thresholds — all evaluated against every event in real time
- Compliance rules deploy as configurable operators within the Job Graph; updates propagate without pipeline restarts
- Full audit trails with lineage: from source event to compliance outcome
- Streamhouse architecture unifies real-time and historical data via Apache Fluss (streaming storage) and Apache Paimon (historical lakehouse)

**Architecture Patterns**

- VERA engine maintains per-account state across millions of concurrent keys with incremental checkpointing
- State snapshots complete without pausing any event — fault tolerance with zero latency impact
- Exactly-once semantics via distributed snapshotting (Chandy-Lamport)
- Event-time processing with watermarks for handling out-of-order events
- Complex Event Processing (CEP) library for pattern-based rule evaluation
- ML inference sub-100ms within pipeline (real-time fraud scoring, anomaly scoring)
- 200+ native connectors and CDC support for integration with existing systems
- Managed infrastructure with auto-scaling based on event throughput and backpressure

**Anomaly Detection Approaches**

- Statistical models and ML scoring executed within the pipeline for anomaly detection
- Pattern detection via Flink CEP: e.g., "Event A followed by Event B within 2 minutes, and Event C has not occurred" → raise alert
- Use cases: credit card fraud, login sequences, multi-failed logins followed by high-value transactions, equipment sensor faults
- SIEM systems: ingest server logs, network device logs, perform real-time CEP to identify attack patterns or policy violations
- Real-time pattern-based alerting enables automated responses within seconds vs. hours/days with batch processing

**Key Use Cases**

- **Fraud Detection**: Detect financial fraud in under 10ms with real-time pattern detection and ML scoring
- **Regulatory Reporting**: Continuous, automated reporting for DORA, Basel III/IV, MiFID II
- **Risk Management**: Real-time exposure tracking, VaR calculation, automated limit breach alerts
- **Customer 360**: Build real-time customer profiles for personalization, churn prediction, and lifetime value scoring inline at stream speed
- **Real-Time ETL + CDC**: Change Data Capture connectors stream database changes (insert/update/delete) to targets in seconds vs. hours

---

### Relevance to CA-DQStream

- **Scale validation**: Both Uber M3 (8.5B points/sec) and Ververica (6.9B records/sec) demonstrate that stream processing at extreme scale is operationally viable, setting a performance ceiling for CA-DQStream evaluation benchmarks
- **Architecture alignment**: The columnar + block-structured storage, lazy evaluation, and per-key state management patterns used at Uber directly inform how CA-DQStream could structure its scoring engine for memory efficiency
- **Data quality in finance**: The Ververica fintech case makes explicit the business cost of delayed/inaccurate data ($4.7B in fines) — providing concrete motivation for CA-DQStream's quality monitoring in streaming pipelines
- **Anomaly detection integration**: Both companies embed statistical/ML anomaly scoring directly into the streaming pipeline, validating CA-DQStream's approach of in-stream anomaly scoring rather than batch post-processing
- **Exactly-once semantics**: Both Uber and Ververica emphasize exactly-once guarantees as foundational — CA-DQStream should account for this requirement when evaluating streaming data quality under node failures
- **Downsampling trade-off**: Uber's finding that naive averaging hides anomalies directly motivates CA-DQStream's use of LTTB or similar quality-preserving aggregation for long-window quality scoring
- **Compliance as a driver**: MiFID II, KYC, DORA compliance requirements create a concrete regulatory pull for real-time data quality — a potential citation motivation for CA-DQStream's framing
- **Flink as substrate**: Ververica's Apache Flink foundation suggests Flink as a natural deployment target or comparison baseline for CA-DQStream

---

## 4. Concept Drift Detection + Anomaly Detection Methods

### 4.1 Concept Drift Detection Methods

#### Types of Drift

- **Sudden Drift**: Abrupt changes in data distribution due to policy changes, data collection changes, or sudden shifts in user behavior. Requires immediate detection and response.
- **Incremental Drift**: Slow, gradual changes in data distribution over time. Challenging to detect as it accumulates subtly.
- **Seasonal Drift**: Predictable, cyclical changes in data (retail, finance, weather). Follows regular patterns and can be anticipated.
- **Virtual Drift**: Changes in feature distribution without changes in the target concept (covariate shift).
- **Real Drift**: Actual changes in the relationship between features and labels (concept change).

#### Statistical Detection Methods

- **Kolmogorov-Smirnov (KS) Test**: Non-parametric test measuring maximum difference between cumulative distributions. Detects changes in continuous feature distributions using sliding windows.
- **Chi-Square Test**: Catches shifts in categorical features by comparing observed vs. expected frequency distributions.
- **Population Stability Index (PSI)**: Industry-standard metric for monitoring score distribution shifts; thresholds: <0.1 (no change), 0.1-0.25 (moderate), >0.25 (significant drift).
- **Wasserstein Distance / Earth Mover's Distance**: Measures distribution shift for numerical features in streaming contexts.
- **KL Divergence**: Quantifies information loss when approximating one distribution with another.

#### Tools & Platforms for Drift Detection

- **Evidently AI**: Open-source Python library for data/ML monitoring. Pre-built tests for data drift, concept drift, and prediction drift. Integrates with Kafka, Airflow, MLflow, Metaflow. Supports tabular and text data with custom thresholds.
- **WhyLabs / whylogs**: Lightweight data profiling that generates statistical profiles of ML inputs/outputs without storing raw data. Ideal for privacy-sensitive streaming applications.
- **NannyML**: Specializes in performance estimation without ground truth labels -- critical for streaming where labels arrive with significant delay. Uses confidence-based estimation.
- **Alibi Detect (Seldon)**: Algorithms specifically for online drift detection in streaming, including outlier and adversarial detection for ML security.
- **Amazon SageMaker Model Monitor**: Automated drift detection for deployed models with built-in alerting.
- **Conduktor**: Data quality module enforcing schema contracts at write time; rejects events violating quality expectations.

#### Integration Patterns

- **Sliding Window Comparison**: Reference window vs. current window (e.g., last hour vs. training baseline). 50% overlap common for stability.
- **Apache Flink Stateful Drift Detection**: Flink maintains running statistics (mean, variance, percentiles) per feature across windows. Flink ML (v1.13+) includes built-in operators for online learning and drift adaptation.
- **MLflow Integration**: Log drift metrics, drifted feature counts, p-values as experiment parameters. Compare runs over time with scatter/contour/parallel coordinate plots.
- **Automated Retraining Triggers**: Drift detection -> training job -> validation on holdout -> canary deployment -> full rollout.

#### Governance as Drift Prevention

- **Schema Contracts**: Define structure, types, and valid ranges for events. Block invalid events before they corrupt model inputs (e.g., user_age > 500 rejected at write time).
- **Schema Evolution Policies**: Require backward compatibility; new fields optional, existing fields maintain semantics.
- **Topic-Level Governance**: Gold-tier (strict enforcement) for production models; bronze-tier (raw data) for experimentation. Prevents degraded data reaching critical models.

**Sources**: [Coditation - Drift with Evidently/MLflow](https://www.coditation.com/blog/how-to-detect-drift-with-evidently-and-mlflow), [Conduktor - Model Drift in Streaming](https://www.conduktor.io/glossary/model-drift-in-streaming)

### 4.2 Data Quality in Streaming Pipelines

#### Core Quality Dimensions

- **Completeness**: Missing values and partial datasets lead to inaccurate AI predictions and poor decisions.
- **Consistency**: Inconsistent data formats across systems require extensive preprocessing before analysis.
- **Timeliness**: Batch processing delays insights; streaming enables real-time decision-making.
- **Accuracy**: Data must reflect real-world state to produce reliable analytics.
- **Uniqueness**: Duplicate records inflate metrics and distort analysis.
- **Validity**: Data conforms to defined schema and business rules.

#### Challenges in Streaming Contexts

- **Data Silos**: Critical business data locked in disconnected databases; unified integration prevents isolated views.
- **Delayed Data Ingestion**: Batch processing delays insights, making real-time decisions impossible.
- **Schema Evolution**: Upstream producers add fields, change enums, or alter timestamp formats; changes propagate silently through topics.
- **High Data Volumes**: Real-time processing of millions of events can overload pipelines; sharding/partitioning needed.
- **Data Consistency**: Ensuring consistency across distributed systems requires exactly-once processing and event sourcing patterns.

#### Quality Enforcement Strategies

- **Automated Validation**: Detect missing values, inconsistencies, and errors before data reaches analytics platforms.
- **Real-Time Cleansing and Enrichment**: Transform and validate data as it moves through the pipeline.
- **Lineage Tracking**: Track data origin and transformations for audit and compliance (GDPR, CCPA).
- **Schema Registry**: Centralized schema management ensuring backward compatibility across producers and consumers.
- **Fault Tolerance**: Pipeline continues operating smoothly even when components fail.

#### Real-Time Architecture Patterns

- **Lambda Architecture**: Combines batch and streaming layers for both historical accuracy and low-latency results.
- **Kappa Architecture**: Simplifies to streaming-only, treating all data as streams for reduced complexity.
- **In-Memory Processing**: Frameworks like Apache Flink reduce latency by minimizing unnecessary system hops.
- **Decentralized Data Management**: Manage and process data across distributed, cloud-native data lakes for greater agility.
- **CI/CD for Data Pipelines**: Automated testing, deployment, and monitoring ensure safe pipeline changes.

**Sources**: [Striim - Data Quality for AI Analytics](https://www.striim.com/blog/data-quality-availability-ai-analytics/), [Actian - Streaming Data Pipelines](https://www.actian.com/streaming-data-pipelines/)

### 4.3 Anomaly Detection in Streaming Contexts

#### Approaches

- **Statistical Process Control**: Monitor feature distributions (mean, variance, percentiles) against baselines. Threshold violations flag anomalies.
- **Unsupervised Learning**: Isolation Forest, DBSCAN, LOF for detecting outliers without labeled training data.
- **Supervised Learning**: Models trained on labeled anomaly data for high-precision detection.
- **Online Learning with Adaptive Windows**: Incrementally update detection thresholds as data distributions shift.
- **ADWIN (ADaptive WINdowing)**: Algorithm specifically designed for detecting concept drift and change in data streams.

#### Streaming-Specific Considerations

- **Label Delay Problem**: In fraud detection, ground truth labels may arrive days after predictions. Solutions: confidence-based estimation (NannyML), delayed-feedback modeling.
- **Feature Drift Detection**: Monitor individual input features for distribution shifts; high-dimensional spaces use embeddings or PCA for drift.
- **Prediction Distribution Monitoring**: Track proportion of positive classifications, confidence score distributions, and predictions near decision boundaries.
- **Seasonality-Aware Detection**: Distinguish anomalies from expected periodic patterns (e.g., Black Friday traffic spikes).

#### Architecture Patterns

- **Kafka + Stream Processing**: KafkaConsumer maintains sliding window of recent events; statistical tests run on each window fill.
- **Apache Flink State Backends**: Maintain running statistics per feature across windows without recomputing from raw events.
- **Microservices for Specialized Detection**: Distribute anomaly, drift, and quality detection across independent services for scalability.

#### Benchmarks & Metrics

- **Evidently AI benchmarks**: 1000-event windows with 50% overlap for stable drift signals.
- **KS test p-value < 0.05**: Common threshold for declaring drift in individual features.
- **PSI thresholds**: < 0.1 (stable), 0.1-0.25 (warning), > 0.25 (action required).

**Sources**: [Coditation - Drift with Evidently/MLflow](https://www.coditation.com/blog/how-to-detect-drift-with-evidently-and-mlflow), [Conduktor - Model Drift in Streaming](https://www.conduktor.io/glossary/model-drift-in-streaming), [Actian - Streaming Data Pipelines](https://www.actian.com/streaming-data-pipelines/)

### 4.4 Relevance to CA-DQStream

| Source Technique | CA-DQStream Relevance | Notes |
|---|---|---|
| Evidently AI drift detection (Evidently + MLflow integration) | **High** | CA-DQStream can adopt sliding-window statistical tests with configurable thresholds. Reference/current window paradigm maps directly to context-aware drift detection. |
| KS Test + Chi-Square for feature-level drift | **High** | Core statistical foundation for CA-DQStream's drift detection module. Non-parametric tests suit streaming data where distributional assumptions may not hold. |
| Alibi Detect online drift detection | **Medium** | Provides algorithmic reference for streaming-specific drift detectors; ADWIN-based approaches are directly applicable. |
| NannyML confidence-based estimation | **High** | Solves the label-delay problem -- directly relevant to CA-DQStream's unsupervised data quality assessment when ground truth is unavailable. |
| Schema contracts + write-time validation | **High** | Governance pattern CA-DQStream can implement at the ingestion layer to filter anomalies before they enter the quality scoring pipeline. |
| Flink stateful drift detection (running stats) | **High** | CA-DQStream's stateful windowed quality scoring aligns with Flink's state backend pattern for maintaining feature-level statistics. |
| PSI metric for score distribution monitoring | **Medium** | Useful for monitoring CA-DQStream's own quality score distributions over time. |
| Hybrid Lambda/Kappa architectures | **Medium** | CA-DQStream's reference/current window comparison is architecturally similar to Lambda's batch+streaming combination. |
| ADWIN for concept drift in data streams | **High** | Direct algorithmic match for CA-DQStream's adaptive windowing strategy. Specifically designed for non-stationary streaming data. |

---

## 5a. NYC Taxi Anomaly Detection Literature

This section surveys academic papers and benchmark resources on anomaly detection applied to NYC Taxi data, with focus on streaming methods, concept drift handling, and evaluation methodologies.

### Dataset Overview

**NYC Taxi Dataset (Numenta Anomaly Benchmark)**

- **Source**: NYC Taxi and Limousine Commission (TLC) trip record data
- **Format**: Aggregated taxi passenger counts in 30-minute intervals
- **Time Period**: July 2014 – January 2015 (approximately 6 months)
- **Records**: ~10,320 data points (source), with implementations on 2014 Yellow Taxi Trip Data (millions of individual trip records)
- **Ground Truth**: Five documented anomalies in the NAB dataset:
  - NYC Marathon (November 2, 2014)
  - Thanksgiving (November 27–28, 2014)
  - Christmas Day (December 25–26, 2014)
  - New Year's Day (January 1–2, 2015)
  - January 2015 Blizzard (January 26–28, 2015)
- **Access**: Available via [Numenta/NAB on GitHub](https://github.com/numenta/NAB/tree/master/data/realKnownCause) in the `realKnownCause` folder (known anomaly causes requiring no hand labeling)

---

### Key Papers

#### 1. RRCF vs. Isolation Forest Comparison

**Reference**: "Anomaly Detection on NYC Taxi Data" – K-Lab (RRCF library documentation), 2018
**URL**: https://klabum.github.io/rrcf/taxi.html

**Methods**:
- **Robust Random Cut Forest (RRCF)**: Tree-based anomaly detection using collaborative dispersion (CoDisp) scores computed over shingled rolling windows
- **Isolation Forest** (scikit-learn implementation): Unsupervised isolation-based anomaly detection with configurable contamination parameter
- **Shingling**: Rolling window approach (shingle_size=48) to capture temporal patterns in the time series

**Results**:
- Both methods successfully identify the known anomaly events (holidays, blizzard, NYC Marathon)
- RRCF provides continuous anomaly scoring through CoDisp metric
- Isolation Forest requires contamination parameter set to match expected anomaly proportion

**Key Finding**: RRCF and Isolation Forest produce similar detection patterns for the NYC taxi dataset, with both algorithms effectively flagging periods of unusual passenger demand. The rolling window (shingling) approach is critical for capturing temporal dependencies.

---

#### 2. Contextual Outlier Detection with Correlated Measures

**Full Citation**: Kuo, Y.-H., Li, Z., & Kifer, D. (2018). *Detecting Outliers in Data with Correlated Measures*. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM'18), October 22–26, 2018, Torino, Italy. ACM. https://doi.org/10.1145/3269206.3271798

**Authors**: Yu-Hsuan Kuo, Zhenhui Li, Daniel Kifer (Pennsylvania State University)

**Venue**: CIKM'18 (Conference on Information and Knowledge Management)

**Methods**:
- **Robust Regression Model**: Simultaneously models non-outliers and outliers using domain-specified correlation templates (e.g., trip distance vs. trip time)
- **Contextual/Conditional Outlier Detection**: Distinguishes between behavioral attributes (examined for anomalies) and contextual attributes (defining the context)
- **Bias-Aware Modeling**: Explicitly accounts for how outliers can skew the learned model, addressing a gap in prior contextual outlier detection work

**Dataset Details**:
- NYC Taxi trip-level data with attributes: pickup/dropoff locations, trip distance, trip time, fare amount
- Individual trips analyzed for suspicious patterns (e.g., long distance + low fare, zero displacement + long distance)

**Results**:
- Outperforms five baseline outlier detection algorithms on real-world datasets
- Robust to extremely noisy datasets common in sensor measurements
- Successfully identifies root causes of outliers (e.g., sensors from certain manufacturers producing anomalous readings)

**Concept Drift Handling**: Not explicitly addressed; method focuses on static correlation analysis per time window rather than temporal drift adaptation.

**Key Contribution**: Introduces the insight that outliers can bias the contextual model itself, making non-outliers appear anomalous. The robust regression approach simultaneously fits the model and assigns outlier probabilities.

---

#### 3. Switching Scheme for Incremental Concept Drift

**Full Citation**: Baier, L., Kellner, V., Kühl, N., & Satzger, G. (2020). *Switching Scheme: A Novel Approach for Handling Incremental Concept Drift in Real-World Data Sets*. arXiv:2011.02738 [cs.LG].

**Authors**: Lucas Baier, Vincent Kellner, Niklas Kühl, Gerhard Satzger (Karlsruhe Institute of Technology, Germany)

**Venue**: arXiv preprint (submitted 2020)

**Methods**:
- **Switching Scheme**: Novel drift handling strategy combining retraining and incremental model updates
- **Drift Detection**: Compared ADWIN (Adaptive Windowing), STEPD (Statistical Test of Equal Proportions), and HDDDM (Hellinger Distance-based Drift Detection Method)
- **Baseline Comparisons**: Regular adaptation (periodic retraining) vs. triggered adaptation (retrain on detected drift)

**Dataset Details**:
- NYC Taxi demand data heavily influenced by changing demand patterns over time
- Focus on incremental concept drift as typical for long-horizon deployed systems

**Results**:
- The switching scheme outperforms all baseline approaches on NYC taxi demand forecasting
- Static models degrade over time without drift handling
- Any drift detection strategy outperforms no strategy; differences between strategies are significant

**Concept Drift Handling**:
- ADWIN: Adaptive sliding window size to match different rates of change; partition window observations and compare error rates among subwindows
- STEPD: Monitors recent vs. overall classifier accuracy
- HDDDM: Monitors input feature distributions using Hellinger distance

**Key Finding**: Combining the advantages of periodic retraining and triggered incremental updates outperforms either approach alone. NYC taxi demand is particularly susceptible to incremental drift due to changing patterns (weather, events, economic factors).

---

#### 4. DriftSense: Adaptive Drift Detection with Incremental Hoeffding Trees

**Full Citation**: Rahman, M.M., Mamun, Q., Bewong, M., & Islam, M.Z. (2026). *DriftSense: Adaptive Drift Detection with Incremental Hoeffding Trees for Real-Time Spatial Crowdsourcing*. In Q.V. Nguyen et al. (Eds.), Data Science and Machine Learning (AusDM 2025). Communications in Computer and Information Science, vol 2765. Springer, Singapore. https://doi.org/10.1007/978-981-95-6786-7_7

**Authors**: Md Mujibur Rahman, Quazi Mamun, Michael Bewong, Md Zahidul Islam (Charles Sturt University, Australia)

**Venue**: AusDM 2025 (Australasian Data Mining Conference)

**Methods**:
- **Spatially Localized Entropy-based Drift Detection**: Novel approach incorporating spatial locality into drift detection
- **Model-Aware ADWIN (MA-ADWIN)**: Incorporates internal signals from Adaptive Hoeffding Trees (AHTs) for drift detection
- **False-Signal Filtering**: Mechanism for robust adaptation reducing false alarms
- **Incremental Hoeffding Trees**: Online decision trees for streaming classification

**Dataset Details**:
- Real-world NYC Taxi dataset with injected abrupt, gradual, and mixed drift types
- Yelp dataset for spatial crowdsourcing context

**Results**:
- Up to **25% higher detection accuracy** compared to baselines
- **8–15% reduction in false alarms**
- **20–25% lower computational overhead**
- Effective and lightweight — suitable for deployment in dynamic platforms

**Concept Drift Handling**:
- Directly addresses concept drift with three innovations:
  1. Spatial locality in drift detection (not previously addressed)
  2. Model-aware ADWIN leveraging AHT internal signals
  3. False-signal filtering for robustness

**Key Contribution**: Introduces spatial awareness into drift detection — critical for geospatial data like taxi demand where drift may be localized to specific areas or time periods.

---

#### 5. LSTM Autoencoder for Time Series Anomaly Detection

**Reference Sources**:
- Malhotra, P., et al. (2016). *LSTM-based Encoder-Decoder for Multi-site Anthropogenic Damage Detection*. (Foundational LSTM encoder-decoder for anomaly detection)
- GitHub: anindya-saha/Machine-Learning-with-Python (LSTM Autoencoder for NYC Taxi Rides)

**Methods**:
- **LSTM Autoencoder**: Unsupervised encoder-decoder architecture that learns to reconstruct normal patterns
- **Reconstruction Error**: Anomaly score based on how poorly a data point is reconstructed
- **Temporal Context**: LSTM layers capture long-range dependencies critical for understanding that the same value means different things at different times (e.g., 5,000 passengers at 3 AM vs. 5 PM)

**Dataset Details**:
- NYC Taxi passenger count time series (30-minute intervals)
- Successfully detects five known anomalies: NYC Marathon, Thanksgiving, Christmas, New Year's, Blizzard

**Results**:
- Successfully identifies all five major anomaly events
- Exploits strong periodic patterns: daily rhythms (higher daytime demand) and weekly trends (day-of-week variation)

**Key Strengths**:
- No labeled anomalous data required for training
- Intuitive anomaly scoring via reconstruction error
- Captures complex temporal dependencies in taxi demand patterns

---

### Comparative Summary

| Paper | Method | Dataset Scale | Concept Drift Handling | Key Advantage |
|-------|--------|--------------|------------------------|---------------|
| RRCF/Isolation Forest | Tree-based isolation | ~10K points | None (static windows) | Benchmark comparison; rolling window approach |
| Kuo et al. (CIKM'18) | Robust regression | Millions of trips | None | Contextual outlier detection; root cause identification |
| Baier et al. (arXiv'20) | Switching scheme | Demand time series | ADWIN, STEPD, HDDDM | Combines retraining + incremental updates |
| Rahman et al. (AusDM'26) | DriftSense (MA-ADWIN) | Taxi + Yelp | Spatial entropy + MA-ADWIN | Spatial locality; 25% accuracy improvement |
| LSTM Autoencoder | Deep learning | ~10K points | None | Captures complex temporal dependencies |

---

### Methods by Category

**Statistical / Classical**:
- Interquartile Range (IQR), Median Absolute Deviation (MAD), Local Outlier Factor (LOF)
- Results on NYC taxi: IQR detected 2 anomalies (0.019%), MAD found 1 (0.009%), LOF identified 1,032 (10%)

**Tree-Based Isolation**:
- Isolation Forest (scikit-learn)
- Robust Random Cut Forest (RRCF)
- Adaptive Hoeffding Trees with MA-ADWIN

**Deep Learning**:
- LSTM Autoencoders for time series reconstruction
- Vanilla autoencoders for passenger count anomaly detection

**Ensemble Methods**:
- RRCF forest ensemble with collaborative dispersion scoring
- Streaming Anomaly Detection (PySAD) library implementations

**Regression-Based**:
- Robust regression with simultaneous outlier modeling (Kuo et al.)

---

### Streaming and Real-Time Implementations

1. **Apache Spark + Pub/Sub**: Real-time NYC Taxi stream processing with Isolation Forest
   - Four-stage pipeline: data preparation → model training → anomaly checking → evaluation
   - Features per ride: distance, travel time, meter readings

2. **PySAD Library**: Streaming anomaly detection algorithms for Python
   - Implements multiple streaming AD methods on NYC taxi data

3. **Striim AI Prototype**: Production-ready streaming LSTM encoder-decoder
   - Combines Kafka, Spark, and live dashboard for real-time detection

---

### Relevance to CA-DQStream

- **NAB dataset as benchmark**: The labeled NYC taxi anomalies provide a standard evaluation set for comparing CA-DQStream's anomaly detection accuracy against RRCF and Isolation Forest baselines
- **Rolling window approach**: RRCF's shingling technique (rolling windows of 48 time steps) directly informs CA-DQStream's context window management for time-series quality scoring
- **Concept drift integration**: The Baier et al. switching scheme and DriftSense framework validate that drift detection should be integrated into streaming quality monitoring — CA-DQStream can adopt ADWIN-style adaptive windows
- **Root cause analysis**: Kuo et al.'s method for identifying outlier causes (e.g., sensor malfunction, specific event types) provides methodology for CA-DQStream to correlate quality anomalies with their underlying causes
- **Multi-scale patterns**: LSTM and RRCF both exploit daily/weekly seasonality — CA-DQStream should incorporate periodicity detection into its context-aware quality scoring

---

### Sources

1. K-Lab. (2018). *Anomaly Detection on NYC Taxi Data | RRCF*. https://klabum.github.io/rrcf/taxi.html
2. Kuo, Y.-H., Li, Z., & Kifer, D. (2018). *Detecting Outliers in Data with Correlated Measures*. CIKM'18. https://doi.org/10.1145/3269206.3271798
3. Baier, L., Kellner, V., Kühl, N., & Satzger, G. (2020). *Switching Scheme: A Novel Approach for Handling Incremental Concept Drift in Real-World Data Sets*. arXiv:2011.02738.
4. Rahman, M.M., Mamun, Q., Bewong, M., & Islam, M.Z. (2026). *DriftSense: Adaptive Drift Detection with Incremental Hoeffding Trees for Real-Time Spatial Crowdsourcing*. AusDM 2025. https://doi.org/10.1007/978-981-95-6786-7_7
5. Malhotra, P., et al. (2016). *LSTM-based Encoder-Decoder for Multi-site Anthropogenic Damage Detection*.
6. Numenta. (2018). *NAB (Numenta Anomaly Benchmark)*. https://github.com/numenta/NAB
7. SoftwareMill. (2023). *Isolation Forest Anomaly Detection with Spark and NYC Taxi Data Stream*. https://softwaremill.com/isolation-forest-anomaly-detection-with-spark-and-nyc-taxi-data-stream/

---

## 7. Context-Aware Systems & Spatio-Temporal Anomaly Detection

This section reviews context-aware anomaly detection methods, with emphasis on spatio-temporal data (taxi/rideshare trajectories), zone-based and adaptive-threshold approaches, and concept drift detection for streaming data. These collectively form the methodological foundation for context-aware streaming data quality systems.

---

### 7.1 Context-Aware Trajectory Anomaly Detection (Taxi/Rideshare)

Traditional trajectory anomaly detection treats each path as an independent instance, overlooking how individual behavior evolves over time and how location semantics influence what counts as "normal." Context-aware methods address this by integrating spatial, temporal, and individual behavioral signals.

#### 7.1.1 BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection (2025)

> **Xie, J., Kim, J., Chiang, Y.-Y., Zhao, L., & Shafique, K.** (2025). *BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data*. The 2nd ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection (GeoAnomalies '25). ACM. [doi:10.1145/3764914.3770888](https://doi.org/10.1145/3764914.3770888)

**Key Insight:** The paper identifies two critical limitations in prior work: (1) multi-scale spatial semantics (e.g., POI types, neighborhood structure) are underexploited — most frameworks rely on trajectory reconstruction accuracy but miss *location-semantic anomalies* such as an office worker visiting industrial facilities during business hours; and (2) individualized behavior modeling is insufficient — population-level patterns incorrectly flag legitimate personal deviations (e.g., night-shift workers commuting at midnight).

**Method Proposed:** BeSTAD is an unsupervised framework built on VAMBC (Variational Autoencoder for Mobility Behavior Clustering). It operates in three stages:

1. **Multi-scale spatial semantics extraction** — constructs buffers at 500 m, 1000 m, and 2000 m around each staypoint, counts 13 OSM feature types (buildings, landuse, transport, POIs, water, natural areas), indexed via H3 spatial grid (resolution 10, ~15.3 km² per cell).
2. **Individual-level trip behavior clustering** — fuses temporal (time-of-day, day-of-week, weekend indicator, cyclic encoding) and spatial features through a learnable projection layer, then passes through VAMBC (VAE + negative entropy loss + center loss) to assign each trip to one of K=6 behavioral clusters. This decomposes the latent representation into cluster-specific **z^c** and individualized bias **z^b**.
3. **Cross-period behavioral comparison** — for each individual, constructs a behavioral profile from the past period containing: cluster distribution **D_a**, transition matrix **M_a**, dominant cluster **C_a**, entropy **H_a**, and trip count. Then compares the test-period profile against the training profile across six dimensions: Jensen-Shannon divergence of cluster distributions (w₁=0.25), dominant cluster change (w₆=0.10), new cluster emergence (w₂=0.20), transition pattern change via Frobenius norm (w₃=0.15), entropy change (w₄=0.15), and frequency change (w₅=0.15). A cluster semantic alignment step (nearest-neighbor in latent z-mean space) ensures consistent cluster interpretation across time periods.

**Results:** Evaluated on NUMOSIM (synthetic Los Angeles mobility dataset, 6,000 sampled individuals). BeSTAD achieves **AUROC = 0.775, AP = 0.0096**, compared to Context-Aware Trajectory Anomaly Detection (CA-TAD) at AUROC = 0.586, AP = 0.002. Ablation confirms that combining temporal + spatial semantics (Full model: 0.775) outperforms either alone (T-only: 0.699; S-only: 0.757).

**Dataset:** NUMOSIM — large-scale synthetic mobility benchmark generated from real travel survey data, simulating realistic urban movement in Los Angeles with ground-truth anomaly labels per individual.

**Relevance to CA-DQStream:** The cross-period behavioral comparison framework directly parallels CA-DQStream's reference/current window comparison. The six-dimensional behavioral profile comparison (distribution divergence, dominant mode change, new pattern emergence, transition structure change, entropy change, frequency change) provides a template for multi-faceted quality score comparison that goes beyond simple thresholding.

---

#### 7.1.2 ICAD: Interpretable Component-wise Anomaly Detection (2025)

> **Siampou, M. D., et al.** (2025). *ICAD: A Self-Supervised Autoregressive Approach for Multi-Context Anomaly Detection in Human Mobility Data*. Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025), 595–606. [doi:10.1145/3748636.3762774](https://doi.org/10.1145/3748636.3762774) · PMCID: PMC13075516

**Key Insight:** Existing mobility anomaly detection methods assign a single holistic anomaly score per visit, making it impossible to determine whether an anomaly arose from an unusual *location*, an unusual *arrival time*, or an unusual *departure time*. This lack of interpretability is critical in applications (healthcare monitoring, fraud detection) where analysts need to understand *why* something was flagged.

**Method Proposed:** ICAD decomposes each visit into three distinct components: region cell **r_i**, arrival time **t_i^a**, and departure time **t_i^d**. It trains a self-supervised autoregressive model on normal visit sequences using next-visit prediction. The joint probability is decomposed via chain rule:

```
P(v_i | V_{i-1}) = P(r_i | V_{i-1}) × P(t̂_i^a | r_i, V_{i-1}) × P(d̂_i | t_i^a, r_i, V_{i-1})
```

Anomaly scores are computed *per component*:

- **Spatial anomaly score** (S_v^R): Top-k deviation metric — measures how far the predicted region rank is from the actual region.
- **Temporal anomaly scores** (S_v^i^a, S_v^i^d): Novel *mode-margin scoring* using Gaussian Mixture Model (GMM) density — scores based on the gap between observed likelihood and the nearest GMM mode, providing a principled measure of relative temporal deviation rather than absolute likelihoods.
- **Joint score**: S_v = w₁·S_v^R + w₂·S_v^i^a + w₃·S_v^i^d.

**Results:** Evaluated on a large-scale synthetic human mobility dataset. ICAD outperforms prior methods at both visit-level and agent-level anomaly detection. The component-wise scoring enables fine-grained interpretability.

**Dataset:** Large-scale synthetic human mobility dataset (NUMOSIM-based or equivalent); code publicly available at [github.com/USC-InfoLab/ICAD](https://github.com/USC-InfoLab/ICAD).

**Relevance to CA-DQStream:** The component-wise scoring paradigm is directly applicable to multi-dimensional quality scoring — each quality dimension (completeness, timeliness, consistency, validity) can have its own anomaly sub-score, and the weighted combination provides both accuracy and interpretability about *which quality dimension* triggered an alert.

---

#### 7.1.3 TAPS: Real-Time Taxi Spatial Anomaly Detection via Trajectory Prediction (2021)

> **Chen, B., et al.** (2021). Real-time taxi spatial anomaly detection based on vehicle trajectory prediction. *Travel Behaviour and Society*, 34, 100698. [doi:10.1016/j.tbs.2021.100698](https://doi.org/10.1016/j.tbs.2021.100698)

**Key Insight:** Taxi services face significant fraud via route manipulation — unethical drivers take detours, especially when passengers are unfamiliar with the area. Monitoring anomalous trajectories is critical for passenger safety and service quality, but real-time detection requires both accuracy and low latency.

**Method Proposed:** TAPS (Taxi Anomaly detection via Prediction-based detection System) uses a two-stage framework:

1. **Offline training stage**: Train a vehicle trajectory prediction model using *recommended routes* from a navigation platform (normal behavior baseline).
2. **Online detection stage**: For each active taxi, compare its current GPS position against the predicted position (from the trained model) and the origin location. An anomaly is flagged when the deviation between actual and predicted positions exceeds a learned threshold.

The key context variable is the **navigation platform's recommended route** — which encodes the "expected" spatial trajectory given the origin-destination pair. Any significant divergence from this recommended path during the trip constitutes a spatial anomaly.

**Results:** Real-world case study shows TAPS achieves greater Accuracy, Precision, and F1 score compared to two baseline methods for detecting anomalous taxi trajectories.

**Dataset:** Real-world taxi trajectory data (city unspecified; likely Shanghai or Beijing given ECNU affiliation).

**Relevance to CA-DQStream:** The offline-training / online-detection architecture mirrors CA-DQStream's reference window (training) vs. current window (detection) design. The recommended-route baseline is analogous to the "expected quality profile" built from historical data.

---

#### 7.1.4 MTRI: Multi-Scale Temporal Model for Vehicle Trajectory Anomaly Detection (2026)

> **Chen, J., Chen, H., & Lu, H.** (2026). Enhancing Road Safety and Sustainability: A Multi-Scale Temporal Model for Vehicle Trajectory Anomaly Detection in Road Network Interactions. *Sustainability*, 18(2), 597. [doi:10.3390/su18020597](https://doi.org/10.3390/su18020597)

**Key Insight:** Anomalous vehicle trajectories include not just route-level detours but also micro-detours, looping behavior, lingering at low speed, and sudden speed changes — each requiring different detection capabilities. Existing methods suffer from scarce labeled anomaly data, inadequate spatial feature extraction in complex road networks, and limited capability to identify complex compound behaviors.

**Method Proposed:** MTRI (Multi-scale Temporal and Road Network Interaction Anomaly Detection model) has three components:

1. **CL-CD (Contrastive Learning-based Conditional Diffusion Model)**: Generates synthetic anomalous trajectories to address data scarcity. Trained contrastively to distinguish normal vs. anomalous trip embeddings.
2. **UNIM (Urban road Network Interaction Modeling)**: Models trajectory-road network interactions using an Edge-Augmented Heterogeneous Attention Network (EA-HAN) over a heterogeneous graph (intersection nodes, road segment nodes, zone nodes).
3. **LSTAD (Long-Short Temporal Anomaly Detection)**: Captures multi-scale temporal features (short-term speed/acceleration patterns, long-term route-level deviations) for detecting sophisticated anomalies.

**Results:** Evaluated on Porto, Portugal real-world taxi/GPS trajectory dataset. MTRI achieves **AUC-ROC > 0.85** across diverse anomaly types and anomaly proportion scenarios.

**Dataset:** Real-world GPS trajectories from Porto, Portugal.

**Relevance to CA-DQStream:** The heterogeneous graph modeling of road network interactions is conceptually analogous to modeling the structural relationships between data quality dimensions. The multi-scale temporal modeling (short-term vs. long-term features) maps to CA-DQStream's sliding window vs. long-window quality scoring.

---

### 7.2 Zone-Based and Location-Aware Anomaly Detection

#### 7.2.1 ReAD: Regional Anomaly Detection via Dynamic Partition (2020)

> **Luo, H., Meng, C., Wu, B., Zhang, J., Li, T., & Zheng, Y.** (2020). *ReAD: A Regional Anomaly Detection Framework Based on Dynamic Partition*. arXiv:2007.06794. [arXiv:2007.06794](https://ar5iv.labs.arxiv.org/html/2007.06794)

**Key Insight:** Existing regional anomaly detection uses fixed partitions (road-based or grid-based), which suffer from two fundamental problems: **data sparsity** (uneven observation distribution leaves many cells empty or sparse) and **heterogeneity** (cells may contain diverse phenomena that blur anomaly signals). The fix — larger cells — reduces spatial resolution.

**Method Proposed:** ReAD uses a **dynamic region partition** approach that adapts region boundaries based on both spatial proximity and reading similarity. Key steps:

1. **Delaunay triangulation-based spatial clustering**: Parameter-free clustering of locations using Delaunay triangulation, ensuring adjacent spatial coverage.
2. **Reading-based clustering**: Separate clustering of sensor readings at each time step.
3. **Intersection operation**: Final regions are the intersection of location-clusters and reading-clusters, guaranteeing that each region satisfies: (a) adjacent locations, (b) similar readings. This eliminates the need for fixed grid size parameters.
4. **Relative divergence metric**: For each region, compute divergence from surrounding regions — this is context-aware because the same absolute reading value is interpreted relative to neighbors (e.g., high ridership in a city center is normal, but the same value in a suburban area is anomalous).
5. **Two detection modes**: *Weighted approach* (for spatial-only anomalies, using temporal divergence as weight) and *Wavy approach* (for spatio-temporal anomalies, tracking divergence fluctuations over time).

**Results:** Validated on synthetic spatio-temporal datasets and two real-world urban data applications, demonstrating effectiveness and practicality of dynamic partition over fixed-grid and road-based methods.

**Relevance to CA-DQStream:** ReAD's dynamic partition is conceptually analogous to adaptive thresholding per zone/region. Instead of a fixed global anomaly threshold, each region has its own dynamically computed threshold based on local context (neighboring regions' values). This is directly applicable to CA-DQStream's zone-aware quality thresholds — e.g., different data quality expectations for different geographic zones, event types, or user segments.

---

### 7.3 Concept Drift Detection with Adaptive/Autonomous Thresholds

#### 7.3.1 DTD: Autonomous Dynamic Threshold Determination for Concept Drift Detection (AAAI 2026)

> **Lu, P., Lu, J., Liu, A., Yu, E., & Zhang, G.** (2026). *Autonomous Concept Drift Threshold Determination*. AAAI Conference on Artificial Intelligence (AAAI 2026). [arXiv:2511.09953](https://arxiv.org/html/2511.09953v1)

**Key Insight:** All existing drift detection methods treat the detection threshold as a **fixed hyperparameter** — set once to balance false alarms vs. late detection, then applied uniformly across all datasets and all time. This paper proves formally that no single fixed threshold can be universally optimal, and that a dynamically adapting threshold is strictly superior.

**Theoretical Contributions:**

- **Theorem 1**: Perfect detection (zero false alarms, zero delay) may not be optimal for model performance — early adaptation can sometimes hurt if the drift is minor.
- **Theorem 2**: No single fixed threshold can be universally optimal across all data segments.
- **Theorem 3 (Dynamic Superiority)**: A dynamic threshold strategy (combining the best threshold from each individual segment) is guaranteed to outperform any fixed threshold applied uniformly.

**Method Proposed:** DTD (Dynamic Threshold Determination) algorithm:

1. When a drift alarm fires, run three candidate models in parallel for several time steps: (a) the old model with delayed adaptation (detection was too late), (b) the new model with immediate adaptation (correct timing), (c) the old model with no adaptation (false alarm).
2. Compare their performance during the comparison window.
3. Use the best performer's label to adjust the threshold for future detections: if (a) won → lower threshold (more sensitive), if (b) won → keep current threshold, if (c) won → raise threshold (less sensitive, reduce false alarms).

**Results:** On the real-world Airline dataset, HDDM-W raised 36 alarms with only 48.64% accuracy. DTD-enhanced HDDM-W raised only 3 alarms with **58.31% accuracy** — a ~10 percentage point improvement. Extensive experiments on synthetic and real-world datasets (image and tabular) confirm substantial performance gains.

**Relevance to CA-DQStream:** This paper provides the formal justification for CA-DQStream's adaptive threshold approach. Instead of hard-coding quality thresholds, DTD's framework suggests that thresholds should evolve based on detected performance (quality score behavior) after each concept change. The three-way comparison approach (too early / correct / false alarm) maps to CA-DQStream's ability to distinguish between genuine quality degradation vs. seasonal variation vs. false positive quality anomalies.

---

#### 7.3.2 ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection (KAIS 2025)

> **Assis, D. N., & Souza, V. M. A.** (2025). ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams. *Knowledge and Information Systems*, 67, 10005–10034. [doi:10.1007/s10115-025-02523-1](https://doi.org/10.1007/s10115-025-02523-1)

**Key Insight:** ADWIN (Adaptive Windowing) is the state-of-the-art performance-monitoring drift detector, but it requires **labeled data** to compute classification accuracy — which is costly or impractical in many streaming scenarios. ADWIN-U extends ADWIN to the unsupervised setting by replacing the accuracy-based comparison with an unsupervised divergence measure between two adaptive windows.

**Method Proposed:** ADWIN-U:

1. Maintains two adaptive windows: a **reference window** (older, stable data) and a **current window** (recent data).
2. When the statistical divergence between the two windows exceeds an adaptive threshold, a drift alarm is raised.
3. The unsupervised divergence can be based on distribution statistics (mean, variance) or predictions from a supervised model when labels are available.

**Novel Metric:** Balanced Accuracy by the Amount of Requested Labeled Data (BAR) — captures the trade-off between detection accuracy and the proportion of labeled data needed for model updates, favoring detectors with low false alarm rates that minimize label dependency.

**Results:** Outperforms the supervised ADWIN across multiple real-world domains when labeled data is scarce or unavailable. The unsupervised adaptation maintains detection effectiveness without requiring continuous label availability.

**Relevance to CA-DQStream:** ADWIN-U's reference/current window comparison is architecturally identical to CA-DQStream's core design. The unsupervised divergence measure (without ground truth labels) maps directly to CA-DQStream's quality score comparison — computing divergence between the reference quality profile and the current quality profile, without needing external ground truth data quality labels.

---

### 7.4 Comprehensive Surveys on Deep Learning Anomaly Detection

#### 7.4.1 Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey (2025)

> **Huang, H., Wang, P., Pei, J., Wang, J., Alexanian, S., & Niyato, D.** (2025). *Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey*. arXiv:2503.13195. [arXiv:2503.13195](https://arxiv.org/abs/2503.13195)

**Scope:** 180+ recent studies (2019–2024), covering deep learning-based AD techniques from leading journals (IEEE, ACM, Springer, Elsevier) and top-tier conferences (AAAI, CCS, ICCV).

**Key Taxonomies Covered:**

*By supervision level:*
- **Supervised AD**: Requires fully labeled dataset; best accuracy when labels are available but impractical for rare anomalies.
- **Semi-supervised AD**: Uses mostly unlabeled data with a small labeled subset; converges toward unsupervised when only normal labels are available.
- **Unsupervised AD**: Learns intrinsic structural properties of data; most common in practice due to label scarcity and cost of anomaly labeling.

*By paradigm:*
- **Reconstruction-based** (Autoencoders, VAEs, GANs): Learn to reconstruct normal data; high reconstruction error signals anomaly.
- **Prediction-based** (RNNs, LSTMs, Transformers): Learn to predict next observation; high prediction error signals anomaly.
- **Hybrid**: Combines reconstruction + prediction losses for improved robustness.

*By data type:*
- Time-series AD, image/video AD, tabular AD, text AD, graph AD.
- Temporal vs. non-temporal categorization.

**Key Findings:**

- Deep learning models significantly outperform traditional methods (statistical, distance-based, clustering) on high-dimensional, complex data.
- Reconstruction-based methods (autoencoders, VAEs) are the most widely used for unsupervised AD due to their simplicity and effectiveness.
- Hybrid methods that combine traditional techniques (e.g., clustering, normalizing flows) with deep learning show the best robustness.
- Key challenges remain: data collection cost, computational complexity, explainability, and handling diverse anomaly types.
- Future directions: causal AD, self-supervised AD, foundation model-based AD, real-time streaming AD.

**Relevance to CA-DQStream:** This survey provides the methodological landscape for CA-DQStream's ML-based quality scoring engine. The reconstruction-based paradigm (train on normal quality profiles, flag high reconstruction error) is directly applicable. The temporal AD taxonomy informs the design of time-series quality score monitoring. The hybrid methods section suggests combining statistical quality tests (PSI, KS test) with deep reconstruction models for robustness.

---

### 7.5 Synthesis: Connecting Context-Aware AD to CA-DQStream

The papers reviewed in this section converge on several principles that directly inform the design of context-aware streaming data quality systems:

| Principle | Source Papers | CA-DQStream Implication |
|---|---|---|
| **Reference/current window comparison** is the dominant paradigm for detecting behavioral change | BeSTAD, ICAD, TAPS, ADWIN-U | CA-DQStream's reference window (historical normal) vs. current window (streaming) is well-validated |
| **Multi-dimensional behavioral profiles** outperform single-score comparisons | BeSTAD (6 dimensions), ICAD (3 components), ReAD (spatial + temporal) | Multi-dimensional quality scoring (completeness, timeliness, consistency, validity, accuracy, uniqueness) is necessary |
| **Context-dependent thresholds** (zone-aware, time-aware) outperform fixed global thresholds | ReAD (relative divergence), DTD (dynamic threshold) | CA-DQStream needs adaptive per-zone and per-time-slot quality thresholds, not fixed global cutoffs |
| **Spatial semantics** (POI types, road network structure) enrich location-aware anomaly detection | BeSTAD (OSM features, H3 indexing), MTRI (heterogeneous road graph) | Geographic/demographic context can enrich quality scoring — different quality expectations for different user segments, device types, or geographic regions |
| **Interpretability** through component-wise scoring is essential for operational adoption | ICAD (spatial vs. temporal decomposition), ReAD (regional divergence) | CA-DQStream must provide per-dimension anomaly scores so operators understand *which quality dimension* triggered an alert |
| **Adaptive windows** that evolve with data are necessary for non-stationary streaming environments | ADWIN-U, DTD | CA-DQStream's reference window must be updated as the data distribution changes; stale reference windows become misleading |
| **Synthetic data augmentation** addresses the fundamental challenge of label scarcity for anomalies | MTRI (CL-CD diffusion model) | CA-DQStream can use generative models to create synthetic quality anomaly scenarios for threshold calibration and system testing |
| **Self-supervised learning** avoids the need for labeled anomaly data | ICAD (next-visit prediction), BeSTAD (VAE clustering) | CA-DQStream can train quality scoring models on normal data without requiring labeled quality anomaly examples |

---

### References

```bibtex
@inproceedings{bestad2025,
  author    = {Xie, Junyi and Kim, Jina and Chiang, Yao-Yi and Zhao, Lingyi and Shafique, Khurram},
  title     = {BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection (GeoAnomalies '25)},
  pages     = {1--12},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3764914.3770888}
}

@inproceedings{icad2025,
  author    = {Siampou, Maria Despoina and others},
  title     = {ICAD: A Self-Supervised Autoregressive Approach for Multi-Context Anomaly Detection in Human Mobility Data},
  booktitle = {Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025)},
  pages     = {595--606},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3748636.3762774},
  pmcid     = {PMC13075516}
}

@article{taps2021,
  author  = {Chen, Bingkun and others},
  title   = {Real-time taxi spatial anomaly detection based on vehicle trajectory prediction},
  journal = {Travel Behaviour and Society},
  volume  = {34},
  pages   = {100698},
  year    = {2021},
  doi     = {10.1016/j.tbs.2021.100698}
}

@article{mtri2026,
  author  = {Chen, Juan and Chen, Haoran and Lu, Hongyu},
  title   = {Enhancing Road Safety and Sustainability: A Multi-Scale Temporal Model for Vehicle Trajectory Anomaly Detection in Road Network Interactions},
  journal = {Sustainability},
  volume  = {18},
  number  = {2},
  pages   = {597},
  year    = {2026},
  doi     = {10.3390/su18020597}
}

@article{read2020,
  author  = {Luo, Huaishao and Meng, Chuishi and Wu, Bowen and Zhang, Junbo and Li, Tianrui and Zheng, Yu},
  title   = {ReAD: A Regional Anomaly Detection Framework Based on Dynamic Partition},
  journal = {arXiv preprint},
  year    = {2020},
  note    = {arXiv:2007.06794},
  url     = {https://ar5iv.labs.arxiv.org/html/2007.06794}
}

@inproceedings{dtd2026,
  author    = {Lu, Pengqian and Lu, Jie and Liu, Anjin and Yu, En and Zhang, Guangquan},
  title     = {Autonomous Concept Drift Threshold Determination},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)},
  year      = {2026},
  note      = {arXiv:2511.09953},
  url       = {https://arxiv.org/html/2511.09953v1}
}

@article{adwinu2025,
  author  = {Assis, D. N. and Souza, V. M. A.},
  title   = {ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams},
  journal = {Knowledge and Information Systems},
  volume  = {67},
  pages   = {10005--10034},
  year    = {2025},
  doi     = {10.1007/s10115-025-02523-1}
}

@article{survey2025,
  author  = {Huang, Haoqi and Wang, Ping and Pei, Jianhua and Wang, Jiacheng and Alexanian, Shahen and Niyato, Dusit},
  title   = {Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {arXiv:2503.13195},
  url     = {https://arxiv.org/abs/2503.13195}
}
```

```markdown
Huang, H., Wang, P., Pei, J., Wang, J., Alexanian, S., & Niyato, D. (2025). Deep learning advancements in anomaly detection: A comprehensive survey. *arXiv:2503.13195*. https://arxiv.org/abs/2503.13195

Assis, D. N., & Souza, V. M. A. (2025). ADWIN-U: Adaptive windowing for unsupervised drift detection on data streams. *Knowledge and Information Systems*, 67, 10005–10034. https://doi.org/10.1007/s10115-025-02523-1

Chen, J., Chen, H., & Lu, H. (2026). Enhancing road safety and sustainability: A multi-scale temporal model for vehicle trajectory anomaly detection in road network interactions. *Sustainability*, 18(2), 597. https://doi.org/10.3390/su18020597

Chen, B., et al. (2021). Real-time taxi spatial anomaly detection based on vehicle trajectory prediction. *Travel Behaviour and Society*, 34, 100698. https://doi.org/10.1016/j.tbs.2021.100698

Luo, H., Meng, C., Wu, B., Zhang, J., Li, T., & Zheng, Y. (2020). *ReAD: A regional anomaly detection framework based on dynamic partition*. arXiv:2007.06794. https://ar5iv.labs.arxiv.org/html/2007.06794

Lu, P., Lu, J., Liu, A., Yu, E., & Zhang, G. (2026). Autonomous concept drift threshold determination. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*. arXiv:2511.09953. https://arxiv.org/html/2511.09953v1

Siampou, M. D., et al. (2025). ICAD: A self-supervised autoregressive approach for multi-context anomaly detection in human mobility data. In *Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025)* (pp. 595–606). ACM. https://doi.org/10.1145/3748636.3762774

Xie, J., Kim, J., Chiang, Y.-Y., Zhao, L., & Shafique, K. (2025). BeSTAD: Behavior-aware spatio-temporal anomaly detection for human mobility data. In *Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection (GeoAnomalies '25)*. ACM. https://doi.org/10.1145/3764914.3770888
```

- Most existing tools focus on ML model drift, not raw data quality. CA-DQStream can position itself as complementary -- measuring *data quality* signals upstream of model drift.
- The combination of multi-dimensional quality scoring (completeness, timeliness, consistency, validity) with context-aware drift detection is underexplored in existing tools.
- Real-time anomaly detection with ADWIN-style adaptive windows and unsupervised confidence estimation (NannyML approach) provides the methodological blueprint for CA-DQStream's core engine.

---

## 5b. Baseline Methods for Streaming Anomaly Detection

This section surveys benchmark studies and key baseline methods for anomaly detection in streaming data, with emphasis on methods suitable for evaluating a Context-Aware Data Quality (CA-DQStream) system over streaming taxi data.

---

### 5b.1 Key Benchmark Studies

#### 5b.1.1 SCAR Benchmark (2024/2025)

**Full Citation:**

Ma, Y., et al. (2025). Revisiting streaming anomaly detection: Benchmark and evaluation. *Artificial Intelligence Review*, 58, 8. Springer. https://doi.org/10.1007/s10462-024-10995-w

**Summary:**

This is the most comprehensive benchmark specifically targeting streaming anomaly detection (as opposed to static or time-series anomaly detection). The authors:

- Propose **SCAR** (Streaming data generator with Customizable Anomalies and concept dRifts), a dataset generator capable of synthesizing streaming data with diverse anomaly types (global vs. local) and concept drift types (sudden, gradual, incremental).
- Evaluate **9 existing streaming algorithms** and **4 adapted static baselines** (iForest, LOF, iNNE, IDK) using a generic **reconstruction strategy** (sliding window with bidirectional scoring) across **76 synthesized datasets** and 74 manipulated real-world datasets.
- Use **AUC-ROC** as the primary metric, evaluated **per-segment** (within stable distribution windows) rather than over the entire stream, addressing a critical flaw in prior benchmarks where incompatible anomaly scores from different model versions are ranked together.

**Methods Compared:**

- *Streaming algorithms*: STORM, HS-Trees, iForestASD, LODA, RRCF, RS-Hash, xStream, MStream, MemStream
- *Adapted static baselines*: iForests, LOFs, iNNEs, IDKs (with reconstruct/sliding-window update strategy)

**Key Findings:**

1. **No single algorithm dominates**: No streaming algorithm statistically outperforms all others across all dataset types (Friedman-Nemenyi test, p ≤ 0.05).
2. **Adapted static baselines are competitive**: The four reconstruction-based baselines (iForests, LOFs, iNNEs, IDKs) ranked in the top four positions, significantly outperforming MStream and LODA on the evaluated datasets.
3. **Global vs. local anomaly detection**: Most algorithms excel at detecting **global anomalies** (points far outside normal data regions). However, **local anomalies** (points that deviate within dense sub-regions) remain challenging -- only **iNNE** and **IDK** (density-based) handle local anomalies effectively. Tree-based (iForest, HS-Trees) and distance-based algorithms struggle with local anomalies.
4. **Concept drift is manageable with timely updates**: When models are updated frequently (reconstruction strategy), different concept drift types (sudden, gradual, incremental) do not substantially degrade detection performance -- provided the update cadence is fine enough.
5. **MemStream underperforms on high-dimensional data** (e.g., image datasets like COIL-20), while LODA struggles on both synthetic and image datasets.
6. **Model update strategy matters more than algorithm choice**: The reconstruction strategy (periodic retraining on latest window) is a surprisingly strong baseline that outperforms many specialized streaming algorithms.

**Concept Drift Handling:**

- SCAR defines four drift types: **sudden** (instantaneous distribution change), **gradual** (blended mixing over a period), **incremental** (smooth parameter drift), and transition changes.
- iForestASD uses a **conditional update**: it detects drift by monitoring outlier rate changes in the sliding window; when the outlier rate drops below a threshold, it rebuilds the iForest model.
- ARCUS uses **anomaly score distribution comparison** between consecutive batches to trigger model updates.
- The SCAR benchmark demonstrates that **timely model updates** (reconstruction strategy) effectively handle all drift types without requiring explicit drift detection -- a practical finding for CA-DQStream.

**Relevance to CA-DQStream:**

- SCAR's evaluation methodology (per-segment AUC-ROC) is directly applicable to CA-DQStream's evaluation protocol.
- The finding that **adapted static baselines are competitive** suggests that CA-DQStream's context-aware scoring can be benchmarked against iForest and LOF adapted with sliding windows -- without requiring specialized streaming algorithms.
- The emphasis on **local anomaly detection** for multi-dimensional quality scoring aligns with CA-DQStream's goal of detecting quality anomalies within contextual subgroups (e.g., fare anomalies within specific time-of-day zones).

---

#### 5b.1.2 ADBench (2022)

**Full Citation:**

Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 15542–15563.

**Summary:**

ADBench is the most comprehensive anomaly detection benchmark covering **30 algorithms** across **57 datasets** (47 existing + 10 new from CV/NLP domains). While not streaming-specific, it provides the foundational comparison framework for tabular anomaly detection.

**Methods Compared (30 total):**

- *Unsupervised (14)*: OC-SVM, Kernel PCA, LOF, kNN, HBOS, COPOD, ECOD, IForest, KDE, CBLOF, F-EA, DeepSVDD, DAGMM, Isolation Forest
- *Semi-supervised (7)*: DeepSAD, DevNet, and others using partial labels
- *Supervised (9)*: XGBoost, LightGBM, CatBoost, Random Forest, MLP, ResNet (adapted for tabular), FT-Transformer

**Performance Metrics:**

- **AUC-ROC** (Area Under ROC Curve)
- **AUCPR** (Area Under Precision-Recall Curve)
- **Critical Difference diagrams** (Wilcoxon-Holm statistical test)

**Key Findings:**

1. **No single unsupervised algorithm is statistically best**: Across 57 datasets, no unsupervised method significantly outperforms others (critical difference diagram). This confirms the "no free lunch" nature of anomaly detection.
2. **Semi-supervised methods excel with limited labels**: With only **1% labeled anomalies**, most semi-supervised methods outperform the best unsupervised method. With **5% labels**, semi-supervised methods achieve median AUC-ROC of **80.95%** vs. **60.84%** for fully-supervised methods.
3. **Ensemble tree methods (XGBoost, CatBoost) are strong**: Even without domain-specific AD design, they achieve competitive performance, particularly when some labels are available.
4. **Deep learning unsupervised methods underperform**: DeepSVDD and DAGMM perform worse than shallow methods without label guidance -- more hyperparameters and harder to tune.
5. **Fastest methods**: HBOS, COPOD, ECOD, and NB (treating each feature independently) are the fastest; XGBOD, ResNet, and FT-Transformer are the slowest.
6. **Anomaly type matters**: The benchmark simulates four anomaly types -- **local** (GMM covariance scaling), **global** (uniform distribution), **dependency** (Vine Copula independence), and **clustered** (scaled GMM clusters) -- and finds that different algorithms perform best for different types.

**Relevance to CA-DQStream:**

- ADBench's finding that **no unsupervised algorithm dominates** justifies CA-DQStream's multi-algorithm comparison approach.
- The **semi-supervised advantage** with minimal labels suggests that even partial ground-truth quality labels (e.g., human-verified quality assessments) can substantially boost CA-DQStream's scoring accuracy.
- The **runtime ordering** (HBOS/COPOD/ECOD fastest) provides guidance for CA-DQStream's real-time scoring layer -- these could serve as lightweight first-pass detectors before more expensive context-aware models.

---

#### 5b.1.3 Self-Supervised Anomaly Detection Survey (2022)

**Full Citation:**

Hojjati, H., Ho, T. K. K., & Armanfard, N. (2025). Self-Supervised Anomaly Detection: A Survey and Outlook. *Neural Networks*, 170, 1083–1096. https://doi.org/10.1016/j.neunet.2023.10.034

**Summary:**

A comprehensive survey of self-supervised learning (SSL) approaches for anomaly detection. SSL methods have emerged as state-of-the-art by learning representations through proxy tasks (e.g., geometric transformations, colorization) without requiring labeled data.

**Key Findings:**

1. **SSL outperforms traditional unsupervised methods**: By learning task-agnostic representations from unlabeled data, SSL-based AD methods significantly outperform classical approaches (KDE, OC-SVM, Isolation Forest).
2. **Proxy task design is critical**: The choice of self-supervised proxy task (e.g., predicting rotation angle, patch location, contrast) determines the quality of learned representations for anomaly detection.
3. **Two high-level categories**: SSL methods are divided into those requiring **negative samples** (contrastive methods) and those not requiring them (predictive methods). Both categories show strong AD performance.
4. **Applicable across data types**: The survey covers tabular, image, time-series, and graph data -- relevant for CA-DQStream's multi-modal taxi data (GPS coordinates, timestamps, fares, passenger counts).

**Relevance to CA-DQStream:**

- Self-supervised proxy tasks (e.g., predicting the time-of-day or day-of-week from partial features) could be used to learn richer contextual representations for quality scoring.
- SSL methods are particularly valuable when labeled anomalies are scarce but unlabeled normal data is abundant -- matching CA-DQStream's scenario where ground-truth quality labels are expensive to obtain.

---

#### 5b.1.4 Isolation Forest In-Depth Study (2022)

**Full Citation:**

Chabchoub, Y., Togbe, M. U., Boly, A., & Chiky, R. (2022). An in-depth study and improvement of Isolation Forest. *IEEE Access*, 10, 10219–10247. https://doi.org/10.1109/ACCESS.2022.3144425

**Summary:**

A comprehensive empirical study of Isolation Forest (iForest), including parameter sensitivity analysis, extended variants (EIF, MVIForest), and comparisons on both real and synthetic datasets.

**Key Findings:**

1. **Linear time complexity**: IForest achieves O(tψ log ψ) training complexity, making it suitable for large-scale streaming applications.
2. **Key parameters**: Number of trees (t), sample size (ψ), and decision threshold significantly impact performance. Optimal ψ is typically 256 for most datasets.
3. **MVIForest (Majority Voting IForest)**: A new variant using per-tree majority voting instead of averaging anomaly scores, achieving **similar accuracy with shorter execution time**.
4. **Extended IForest (EIF)**: Improves split value selection to reduce false positives, but at increased computational cost.
5. **iForest struggles with local anomalies**: In high-density regions, iForest cannot isolate local anomalies because they remain in dense regions alongside normal data.
6. **Including anomalies in training is beneficial**: Contrary to early assumptions, training with a small number of known anomalies (up to 5% contamination) can improve detection.

**Relevance to CA-DQStream:**

- IForest's linear complexity and low memory requirement make it a practical baseline for CA-DQStream's real-time scoring.
- The **MVIForest** variant offers a speed-accuracy trade-off suitable for high-throughput taxi data streams.
- The finding about **local anomalies** is directly relevant: fare anomalies within specific time windows or zones are local anomalies that iForest alone may miss -- motivating CA-DQStream's context-aware approach.

---

#### 5b.1.5 Streaming Anomaly Detection Evaluation Study (2023)

**Full Citation:**

Iglesias Vázquez, F., et al. (2023). Anomaly detection in streaming data: A comparison and evaluation study. *Expert Systems with Applications*, 213, Part C, 119296.

**Summary:**

An evaluation of 8 unsupervised outlier detection methods for streaming data using 180 synthetic datasets and 4 real-world datasets, focusing on the challenges of space geometries, nonstationarity, concept drift, memory span, and the definition of outlierness.

**Methods Compared:**

- LOF, HBOS, Isolation Forest, One-Class SVM, DBSCAN-based methods, and streaming-adapted variants

**Key Findings:**

1. **Nonstationarity is the dominant challenge**: Methods that do not adapt to concept drift show significant performance degradation over time.
2. **Memory span affects performance**: Methods with appropriate memory management (sliding windows, reservoir sampling) outperform both memory-limited and memory-heavy approaches.
3. **Definition of outlierness matters**: Different algorithms optimize for different outlier definitions (distance-based, density-based, isolation-based), leading to complementary strengths.
4. **Ensemble methods improve robustness**: Combining multiple base detectors reduces variance and improves robustness to concept drift.

**Relevance to CA-DQStream:**

- The emphasis on **memory management** directly informs CA-DQStream's window sizing strategy for quality scoring.
- The finding that **no single outlier definition is optimal** supports CA-DQStream's multi-dimensional quality scoring across completeness, timeliness, consistency, and validity.
- Ensemble approaches provide a blueprint for combining multiple quality signal detectors in CA-DQStream.

---

### 5b.2 Summary: Recommended Baseline Methods for CA-DQStream

Based on the benchmark literature above, the following methods are recommended as baselines for evaluating CA-DQStream on streaming taxi data:

| Method | Category | Strengths for Taxi Data | Weaknesses |
|--------|----------|------------------------|------------|
| **Isolation Forest** (iForest) | Isolation-based | Fast, linear complexity, good for global anomalies | Struggles with local anomalies in dense regions |
| **MVIForest** | Isolation-based | Same accuracy as iForest, faster execution | Newer, less community tooling |
| **LOF** (Local Outlier Factor) | Density-based | Excellent for local anomalies | O(n²) complexity, memory-intensive |
| **HBOS** (Histogram-based Outlier Score) | Statistical | Extremely fast, interpretable | Treats features independently, misses correlations |
| **COPOD** (Copula-based Outlier Detection) | Statistical | Strong on correlated features, fast | Less effective on very high-dimensional data |
| **ECOD** (Empirical CDF-based) | Statistical | Fast, good for mixed-type data | Assumes empirical CDF sufficiency |
| **MemStream** | Deep learning | Memory-guarded, handles concept drift | Requires training, slower than statistical methods |
| **ARCUS** | Deep learning | Conditional updates for drift, autoencoder-based | Hyperparameter-sensitive, complex tuning |
| **iForestASD** | Streaming-adapted iForest | Built-in concept drift detection | Limited to global anomaly detection |

### 5b.3 Performance Metrics from Benchmarks

The reviewed benchmarks consistently use the following metrics for streaming anomaly detection evaluation:

| Metric | Full Name | Notes |
|--------|-----------|-------|
| **AUC-ROC** | Area Under Receiver Operating Characteristic Curve | Primary metric in SCAR and ADBench. Robust to class imbalance when anomaly rate is low. |
| **AUCPR** | Area Under Precision-Recall Curve | More informative than AUC-ROC when anomaly rate is very low (<5%). |
| **Precision@k** | Precision at top-k | Useful when only the top-k most anomalous points are reviewed. |
| **Recall@k** | Recall at top-k | Complements Precision@k for operational evaluation. |
| **F1 Score** | Harmonic mean of Precision and Recall | Balances false positive and false negative costs. |
| **Detection Delay** | Time from anomaly occurrence to detection | Critical for streaming/fraud scenarios. |
| **False Positive Rate** | Proportion of normal points incorrectly flagged | High FPR leads to alert fatigue in operations. |

### 5b.4 Concept Drift Types and Handling Strategies

From the SCAR benchmark, the following concept drift types are most relevant for streaming taxi data:

| Drift Type | Description | Example in Taxi Data | Recommended Handling |
|-----------|-------------|---------------------|-------------------|
| **Sudden** | Instantaneous distribution change | New airport route opening, sudden policy change | Immediate model retraining on recent window |
| **Gradual** | Slow blending between distributions | Seasonal tourism trends | Continuous model updates (reconstruction strategy) |
| **Incremental** | Smooth parameter drift | Slowly increasing average fares due to inflation | Incremental/online learning updates |
| **Recurring** | Previously seen distributions return | Holiday vs. weekday patterns | Maintain concept memory, enable concept switching |

**Recommended Drift Handling for CA-DQStream:**

1. **Sliding window with reconstruction**: Periodically retrain quality scoring models on the most recent window of data (as demonstrated by the SCAR baseline methods).
2. **Per-segment evaluation**: Evaluate quality scores within stable distribution windows, not over the entire stream, to avoid mixing incompatible scores.
3. **Drift detection triggers**: Use PSI or KS test to trigger model updates when distribution shift exceeds thresholds.
4. **Multi-window ensemble**: Maintain models trained on different time horizons (last hour, last day, last week) and weight their scores based on recent prediction confidence.

---

### 5b.5 Key Takeaways for CA-DQStream Evaluation

1. **Benchmark against adapted static methods**: The SCAR benchmark shows that iForest and LOF adapted with sliding-window reconstruction strategies rank among the top performers, making them appropriate baselines for CA-DQStream.

2. **Evaluate both global and local anomalies**: CA-DQStream should measure detection performance separately for global anomalies (e.g., negative fares, impossible GPS coordinates) and local anomalies (e.g., unusually high fares within a specific time-of-day and zone).

3. **Use per-segment AUC-ROC for evaluation**: Following SCAR's methodology, evaluate quality scoring performance within stable contextual windows rather than over the entire stream.

4. **Concept drift is not optional**: Taxi data exhibits clear seasonal, day-of-week, and event-driven drifts. CA-DQStream's evaluation must include scenarios with concept drift to measure adaptation effectiveness.

5. **Consider semi-supervised enhancement**: Even a small number of human-verified quality labels (1-5%) can substantially improve scoring accuracy -- worth incorporating in CA-DQStream's evaluation protocol if labels become available.

6. **Runtime matters for real-time operation**: HBOS, COPOD, ECOD, and MVIForest offer the best speed-accuracy trade-offs for real-time quality scoring at scale.

---

**Sources:**

- Ma, Y., et al. (2025). Revisiting streaming anomaly detection: Benchmark and evaluation. *Artificial Intelligence Review*, 58, 8. Springer. https://doi.org/10.1007/s10462-024-10995-w
- Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. *NeurIPS 2022*. https://arxiv.org/abs/2206.09426
- Hojjati, H., Ho, T. K. K., & Armanfard, N. (2025). Self-Supervised Anomaly Detection: A Survey and Outlook. *Neural Networks*, 170. https://doi.org/10.1016/j.neunet.2023.10.034
- Chabchoub, Y., Togbe, M. U., Boly, A., & Chiky, R. (2022). An in-depth study and improvement of Isolation Forest. *IEEE Access*, 10, 10219–10247. https://doi.org/10.1109/ACCESS.2022.3144425
- Iglesias Vázquez, F., et al. (2023). Anomaly detection in streaming data: A comparison and evaluation study. *Expert Systems with Applications*, 213, 119296. https://doi.org/10.1016/j.eswa.2022.119296
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *ICDM 2008*. https://doi.org/10.1109/ICDM.2008.17
- Bhatia, S., et al. (2022). MemStream: Memory-Based Streaming Anomaly Detection. *WSDM 2022*. https://doi.org/10.1145/3488560.3498501

---

## 6a. MemStream & Memory-Based Anomaly Detection

### Full Citation

Bhatia, S., Jain, A., Srivastava, S., Kawaguchi, K., & Hooi, B. (2022). *MemStream: Memory-Based Streaming Anomaly Detection*. In Proceedings of the ACM Web Conference 2022 (WWW '22), April 25-29, 2022, Virtual Event, Lyon, France. ACM. https://doi.org/10.1145/3485447.3512221

**BibTeX Entry:**
```bibtex
@inproceedings{bhatia2022memstream,
  title={MemStream: Memory-Based Streaming Anomaly Detection},
  author={Bhatia, Siddharth and Jain, Arjit and Srivastava, Shivin and Kawaguchi, Kenji and Hooi, Bryan},
  booktitle={Proceedings of the ACM Web Conference 2022 (WWW '22)},
  pages={1--12},
  year={2022},
  publisher={ACM},
  doi={10.1145/3485447.3512221}
}
```

### Paper Overview

| Attribute | Details |
|-----------|---------|
| **Authors** | Siddharth Bhatia (NUS), Arjit Jain (IIT Bombay), Shivin Srivastava (NUS), Kenji Kawaguchi (Harvard), Bryan Hooi (NUS) |
| **Venue** | WWW '22 (The Web Conference), April 2022, Lyon, France |
| **arXiv Preprint** | arXiv:2106.03837 |
| **GitHub** | https://github.com/Stream-AD/MemStream (92 stars, Apache 2.0) |

### Method Overview

MemStream combines three key components:

1. **Denoising Autoencoder (DAE)**: Extracts robust feature representations by corrupting input with Gaussian noise, forcing the network to capture essential structure of normal data distribution. The encoder produces a compressed representation $z_t$ of input $x_t$.

2. **Memory Module**: A collection of $N$ real-valued $D$-dimensional vectors storing encodings of normal data. Memory is queried to retrieve $K$-nearest neighbors to the current encoding, which are used to calculate anomaly scores.

3. **FIFO Memory Update Policy**: Memory is updated in a First-In-First-Out manner when anomaly scores fall within an update threshold $\beta$. This preserves temporal locality and enables adaptation to changing data distributions.

### Anomaly Scoring Mechanism

For each incoming record $x_t$:
1. The encoder computes normalized representation $z_t = f_\theta(x_t)$
2. Memory is queried for $K$-nearest neighbors $\{ \hat{z}_1^t, \hat{z}_2^t, ..., \hat{z}_K^t \}$
3. $\ell_1$ distances are calculated: $R(z_t, \hat{z}_i^t) = \| z_t - \hat{z}_i^t \|_1$ for all $i \in [1..K]$
4. Final discounted score: $Score(z_t) = \sum_{i=1}^K \gamma^{i-1} R(z_t, \hat{z}_i^t)$ where $\gamma$ is the discount factor
5. Record is added to memory if $Score(z_t) < \beta$

### Key Technical Contributions

1. **Theoretical Bounds on Memory Size**: The paper proves that optimal memory size should be proportional to (data distribution spread) / (speed of concept drift). This provides a principled method for setting memory length hyperparameters.

2. **Memory Poisoning Robustness**: Two architectural design choices prevent anomalous samples from corrupting memory:
   - Discounting mechanism using $K$-nearest neighbor weights
   - Self-correction capability that recovers from "bad" memory states

3. **Quick Retraining**: Allows rapid retraining when stream becomes sufficiently different from initial training data.

### Benchmark Results

MemStream was evaluated on 2 synthetic datasets and 11 real-world datasets, outperforming 11 state-of-the-art streaming anomaly detection baselines:

| Method | KDD99 | NSL-KDD | UNSW-NB15 | CICIDS-DoS | Ion. | Cardio | Sat. | Sat.-2 | Mamm. | Pima | Cover |
|--------|-------|---------|-----------|------------|------|--------|------|--------|-------|------|-------| 
| xStream (KDD'18) | 0.957 | 0.552 | 0.804 | 0.800 | 0.847 | 0.918 | 0.677 | 0.996 | 0.856 | 0.663 | 0.894 |
| MStream (WWW'21) | 0.844 | 0.544 | 0.860 | 0.930 | 0.670 | 0.986 | 0.563 | 0.958 | 0.567 | 0.529 | 0.874 |
| **MemStream** | **0.980** | **0.978** | **0.972** | **0.938** | **0.821** | **0.884** | **0.727** | **0.991** | **0.894** | **0.742** | **0.952** |

*Table: AUC scores (higher is better). MemStream achieves statistically significant improvements (p < 0.001) over baselines.*

**NSL-KDD AUC-PR Results**:
- MemStream: **0.959 ± 0.002** (55 seconds)
- Best baseline (DILOF): 0.822 ± 0.000 (260 seconds)

### Concept Drift Handling

MemStream specifically addresses concept drift through:

1. **FIFO Memory Update**: Maintains temporal contiguity by replacing oldest entries with newest. Memory retains most recent non-anomalous samples from the distribution.

2. **Drift Scenario Coverage**: Evaluated on multiple drift types:
   - Point anomalies
   - Sudden frequency changes
   - Continuous concept drift
   - Sudden mean shifts
   - Periodic patterns

3. **Adaptation Verification**: Assigns high anomaly scores to trend-changing events, then gradually decreases scores as it adapts to the new distribution.

4. **Memory Size Considerations**:
   - Larger memory: Decreases false positives (more likely to have close neighbors for normal samples)
   - Smaller memory: Decreases false negatives (faster adaptation to new patterns)

### Ablation Study Findings

| Component | Tested Values | Best | KDDCUP99 AUC |
|-----------|--------------|------|--------------|
| Memory Update | None, LRU, RR, FIFO | FIFO | 0.980 |
| Feature Extraction | Identity, PCA, IB, AE | AE | 0.980 |
| Memory Length (N) | 128, 256, 512, 1024 | 256 | 0.980 |
| Output Dimension (D) | d/2, d, 2d, 5d | 2d | 0.980 |
| Update Threshold (β) | 1, 0.1, 0.01, 0.001 | 1 | 0.980 |
| KNN Coefficient (γ) | 0, 0.25, 0.5, 1 | 0 | 0.980 |

### Open Source Implementation

- **Repository**: https://github.com/Stream-AD/MemStream
- **License**: Apache 2.0
- **Framework**: Python + PyTorch
- **Datasets Included**: KDDCUP99, NSL-KDD, UNSW-NB15, CICIDS-DoS, ODDS datasets (Ionosphere, Cardio, Satellite, Satimage-2, Mammograph, Pima, ForestCover)
- **Parameters**: `--beta` (update threshold), `--memlen` (memory size), `--dataset`, `--lr`, `--epochs`

**River Integration**: MemStream has been integrated into the [River](https://github.com/online-ml/river) online machine learning library (PR #1748).

### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Memory-augmented scoring** | MemStream's memory module provides a template for CA-DQStream's quality score memory -- storing reference quality profiles for comparison |
| **FIFO adaptation** | FIFO replacement policy aligns with temporal quality tracking in streaming contexts |
| **Self-correction mechanism** | The K-nearest neighbor discounting approach could inspire CA-DQStream's recovery from corrupted quality scores |
| **Theoretical memory bounds** | Provides methodology for CA-DQStream to determine optimal context window sizes |
| **Benchmark validation** | KDDCUP99, NSL-KDD, UNSW-NB15 results provide baseline comparison targets for streaming quality detection |
| **Periodic pattern handling** | Memory size > (period × sampling frequency) enables detection of periodic quality variations |

---

## 9a. ML for NYC Taxi Anomaly Detection (Batch 1)

Papers focusing on taxi/transportation data anomaly detection, NYC taxi datasets, and isolation forest variants for spatial-temporal data.

---

### Paper 9a.1: K-Means-Based Isolation Forest

**Full Citation:**

Karczmarek, P., Kiersztyn, A., Pedrycz, W., & Al, E. (2020). K-Means-based isolation forest. *Knowledge-Based Systems*, 195, 105659. https://doi.org/10.1016/j.knosys.2020.105659

**BibTeX:**

```bibtex
@article{karczmarek2020kmeans,
  title={K-Means-based isolation forest},
  author={Karczmarek, Pawe{\l} and Kiersztyn, Adam and Pedrycz, Witold and Al, Ebru},
  journal={Knowledge-Based Systems},
  volume={195},
  pages={105659},
  year={2020},
  publisher={Elsevier}
}
```

**Method Proposed:**

The K-Means-Based Isolation Forest (K-IF) is a novel enhancement of the classical Isolation Forest algorithm that replaces binary tree splits with multi-branch splits determined by K-Means clustering. At each node during tree construction:

1. An attribute is randomly selected
2. K-Means clustering (with elbow method for optimal k) determines the number of clusters/splits
3. Points are assigned to cluster-based leaves rather than binary divisions
4. The anomaly score is computed as `s(x) = 1 - d(x, cq)/d(cl, cq)` where cq is the cluster center and cl is the cluster limit

This results in trees that are "wider" (more branches per node) but "shallower" (less depth), and scores that are more intuitive—lower scores indicate anomalies, with values near 0 for isolated points.

**Key Results / Metrics:**

- **Dataset**: NYC Taxi Trip Data (2013) — 3,386,426 records with 14 attributes
- **Key Finding**: K-Means-IF significantly outperforms standard IF for:
  - Identifying points outside NYC geographic boundaries as outliers
  - Assigning zero scores to records with missing/invalid values (99.98% vs 99.56%)
  - Providing more differentiated anomaly rankings (12,500 distinct rank values vs ~1,750 for IF)
- **Execution Time**: For geographic-only data (737,462 records, 2 attributes): 7.47s vs 54.69s for standard IF — approximately **7× faster**
- **Notable**: On intermodal transport data, K-Means-IF produces more granular anomaly differentiation while maintaining comparable accuracy

**Dataset Used:**

1. NYC Taxi Trip Data 2013 (3.38M records, 14 features including pickup/dropoff coordinates, trip time, distance)
2. Synthetic 2D datasets (30K–50K points)
3. European intermodal transport data (ship and train transportation)

**Comparison with Baselines:**

- Compared against standard Isolation Forest
- K-Means-IF shows better performance on:
  - Spatio-temporal data with geographic coordinates
  - Data containing missing/invalid values
  - Cases requiring fine-grained anomaly ranking

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **Geospatial anomaly detection** | Directly applicable to detecting anomalous taxi pickup/dropoff locations (e.g., outside NYC boundaries) |
| **Interpretable scoring** | The cluster-based membership scoring provides clearer anomaly interpretation than standard IF |
| **Missing value handling** | Strong performance on records with invalid/missing values aligns with real-world taxi data quality issues |
| **Transportation domain** | Explicit focus on taxi transport data validates method applicability to NYC taxi anomaly detection |
| **Batch processing note** | Original method is batch; needs adaptation for streaming context |

---

### Paper 9a.2: RIFIFI — Revised Isolation Forest for Fraud and Data Inner Structure

**Full Citation:**

Yepmo, V., Smits, G., Lesot, M.-J., & Pivert, O. (2024). Leveraging an Isolation Forest to Anomaly Detection and Data Clustering. *Data & Knowledge Engineering*, 137, 101946. https://doi.org/10.1016/j.knosys.2024.111946

**BibTeX:**

```bibtex
@article{yepmo2024rififi,
  title={Leveraging an Isolation Forest to Anomaly Detection and Data Clustering},
  author={Yepmo, V{\'e}ronne and Smits, Gr{\'e}gory and Lesot, Marie-Jeanne and Pivert, Olivier},
  journal={Data \& Knowledge Engineering},
  volume={137},
  pages={101946},
  year={2024},
  publisher={Elsevier}
}
```

**Method Proposed:**

RIFIFI (Revised Isolation Forest to Identify Fraud and the data Inner structure) is a variant of Isolation Forest that modifies the split selection criterion based on **density**. The key innovation is:

1. **Density-based Split Selection**: A split is discarded if more than η (threshold) fraction of points fall within margin α around the split
2. **Three Leaf Types**: Generates Isolation Nodes (IN), Dense Nodes (DN), and Depth-Limit Nodes (DLN)
3. **Combined Anomaly Scoring**: Integrates separability (from IF) with local density information (from LOF)
4. **Inseparability Index**: Computes pairwise similarity based on co-occurrence in Dense Nodes, enabling clustering

**Key Results / Metrics:**

| Dataset | IF AUC | RIFIFI AUC | Improvement |
|---------|--------|------------|-------------|
| Annthyroid | 0.802 | 0.778 | -0.024 |
| Arrhythmia | 0.765 | **0.822** | **+0.057** |
| Breast | 0.979 | **0.992** | **+0.013** |
| Cover | 0.885 | 0.857 | -0.028 |
| Hbk | 1.000 | 1.000 | 0 |
| Http | 0.999 | 0.997 | -0.002 |
| Ionosphere | 0.847 | 0.832 | -0.015 |
| **Mammography** | 0.610 | **0.843** | **+0.233** |
| Pima | 0.676 | **0.683** | **+0.007** |
| Satellite | 0.705 | 0.685 | -0.020 |
| Shuttle | 0.994 | 0.993 | -0.001 |
| Smtp | 0.891 | 0.866 | -0.025 |
| Wood | 0.903 | **1.000** | **+0.097** |
| **Mean AUC** | **0.85** | **0.88** | **+0.03** |

- **Mean AUC improvement**: +0.03 across 13 datasets
- **Best improvements**: Mammography (+0.233), Wood (+0.097), Arrhythmia (+0.057)
- **Clustering ARI**: Using inseparability index for AHC achieves higher ARI than Euclidean distance on D3, D4, D5 (stretched clusters, subspace clusters)

**Dataset Used:**

- 13 standard anomaly detection datasets from ODDS library (Annthyroid, Arrhythmia, Breast, Cover, Hbk, Http, Ionosphere, Mammography, Pima, Satellite, Shuttle, Smtp, Wood)
- Synthetic 2D/3D datasets for clustering evaluation (D1–D5 with varying cluster configurations)

**Comparison with Baselines:**

- Compared against: IF, Extended IF (EIF), SCIForest, Fair Cut Forest (FCF)
- RIFIFI shows best mean AUC (0.88) across all variants
- Particularly effective for:
  - Data with varying density regions
  - Detecting local anomalies vs. global anomalies
  - Simultaneous anomaly detection and clustering

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **Local anomaly detection** | Better at detecting anomalies relative to local density — relevant for taxi hotspots vs. sparse areas |
| **Density-aware scoring** | Combines isolation with density — could distinguish unusual taxi activity in dense vs. sparse zones |
| **Multi-purpose framework** | Single method for both anomaly detection and clustering — efficient for taxi zone analysis |
| **Interpretability** | Dense Nodes provide insight into normal data structure — helpful for understanding expected taxi patterns |
| **Batch processing** | Original method is batch; streaming adaptation would need incremental split validation |

---

### Paper 9a.3: IForestASD — Isolation Forest for Streaming Data in Scikit-Multiflow

**Full Citation:**

Togbe, M.U., Barry, M., Boly, A., Chabchoub, Y., Chiky, R., Montiel, J., & Tran, V.-T. (2020). Anomaly Detection for Data Streams Based on Isolation Forest using Scikit-multiflow. *International Conference on Computational Science and its Applications (ICCSA 2020)*, pp. 274–289. https://doi.org/10.1007/978-3-030-58802-1_20

**BibTeX:**

```bibtex
@inproceedings{togbe2020iforestasd,
  title={Anomaly Detection for Data Streams Based on Isolation Forest using Scikit-multiflow},
  author={Togbe, Maurras Ulbricht and Barry, Mariam and Boly, Aliou and Chabchoub, Yousra and Chiky, Raja and Montiel, Jacob and Tran, Vinh-Thuy},
  booktitle={International Conference on Computational Science and its Applications (ICCSA 2020)},
  pages={274--289},
  year={2020},
  organization={Springer}
}
```

**Method Proposed:**

IForestASD (Isolation Forest Algorithm for Streaming Data) adapts the original Isolation Forest for streaming contexts using **sliding window** approach:

1. **Window-Based Processing**: Maintains a sliding window W of recent observations
2. **Batch IF on Window**: When window is complete, builds standard Isolation Forest on all data in window
3. **Concept Drift Detection**: Monitors anomaly rate in window; if exceeds threshold u, rebuilds detector
4. **Scoring**: New instances traverse existing trees; anomaly score computed as in standard IF

**Key Results / Metrics:**

| Dataset | HS-Trees F1 | IForestASD F1 | IForestASD Advantage |
|---------|-------------|---------------|----------------------|
| Shuttle (7.15% anomaly) | 0.13–0.17 | **0.64–0.80** | **4–5× better** |
| Forest-Cover (0.96% anomaly) | 0.36–0.55 | 0.22–0.49 | Mixed |
| SMTP (0.03% anomaly) | 0 | **0.34–0.40** | **Detects where HS-Trees fails** |

**Resource Comparison:**

- **Memory**: IForestASD uses ~20× less memory than Half-Space Trees
- **Speed Trade-off**: IForestASD faster with small windows (W<100); slower with large windows (W≥500)
- **Testing Time**: HS-Trees is 100× faster for scoring; IForestASD exponential in window size

**Window Size Recommendations:**

| Priority | Recommended Setting |
|----------|---------------------|
| Best F1 Score | W ≈ 500 |
| Fast Scoring | HS-Trees preferred |
| Balanced (moderate F1 + speed) | W < 100, T = 30–50 |

**Dataset Used:**

- Shuttle (49,097 samples, 9 features, 7.15% anomaly rate)
- Forest-Cover (286,048 samples, 10 features, 0.96% anomaly rate)
- SMTP (95,156 samples, 3 features, 0.03% anomaly rate)

**Comparison with Baselines:**

- Compared against Half-Space Trees (state-of-the-art streaming anomaly detection)
- IForestASD wins on:
  - Datasets with higher anomaly rates (>1%)
  - F1 score metric (harmony of precision/recall)
- HS-Trees wins on:
  - Execution time (especially testing/scoring)
  - Large window sizes
  - Memory efficiency per sample

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **Streaming IF adaptation** | Direct methodology for adapting IF for streaming taxi data |
| **Window size tuning** | Provides empirical guidance on window selection for taxi data streams |
| **Concept drift handling** | Anomaly rate monitoring aligns with detecting regime changes in taxi demand |
| **Framework availability** | Open-source Scikit-Multiflow implementation accelerates reproducibility |
| **Limitations** | Complete model rebuild on drift may be too aggressive for taxi data; incremental updates preferred |

---

### Paper 9a.4: Anomaly Detection in Time-Series — Comparative Survey

**Full Citation:**

Braei, M. & Wagner, S. (2020). Anomaly Detection in Univariate Time-Series: A Survey on the State-of-the-Art. *arXiv:2004.00433*. Technical University of Darmstadt.

**BibTeX:**

```bibtex
@article{braei2020timeseries,
  title={Anomaly Detection in Univariate Time-Series: A Survey on the State-of-the-Art},
  author={Braei, Mohammad and Wagner, Sebastian},
  journal={arXiv preprint arXiv:2004.00433},
  year={2020},
  institution={Technische Universit{\"a}t Darmstadt}
}
```

**Method Proposed:**

This paper provides a comprehensive survey and comparative evaluation of 20 anomaly detection methods across three categories:

1. **Statistical Methods**: AR, MA, ARMA, ARIMA, Exponential Smoothing (SES, DES, TES), PCI, STL, SARIMA
2. **Machine Learning Methods**: KNN, OC-SVM, Isolation Forest, LOF, DBSCAN
3. **Deep Learning Methods**: LSTM-AE, CNN, MLP-AE, DAMP2

**Key Results / Metrics:**

The paper evaluates methods on multiple real-world univariate time-series datasets. Key findings:

- **Statistical Methods** excel when data follows clear statistical patterns (stationarity, seasonality)
- **Machine Learning Methods** (especially IF and OC-SVM) show robust performance across diverse patterns
- **Deep Learning Methods** require larger datasets and longer training; perform best with sufficient data
- **Best performers by category**:
  - Statistical: ARIMA with appropriate differencing
  - ML: Isolation Forest
  - DL: LSTM Autoencoder

**Key Insights for Time-Series Anomaly Detection:**

1. **No universal winner**: Performance depends heavily on data characteristics
2. **Isolation Forest** consistently performs well across datasets without requiring parameter tuning
3. **Window size** is critical — too small misses patterns, too large introduces noise
4. **Threshold selection** (typically 3σ or percentile-based) significantly impacts results

**Dataset Used:**

- Multiple univariate time-series datasets including:
  - Server monitoring data
  - Sensor data
  - ECG/medical time-series
  - Financial time-series
  - Synthetic data with injected anomalies

**Comparison with Baselines:**

- Comprehensive comparison across 20 methods in 3 categories
- Ranking by precision, recall, F1, and computation time
- Isolation Forest identified as best ML method with minimal hyperparameter sensitivity

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **Method selection guidance** | IF confirmed as robust choice for taxi time-series anomaly detection |
| **Window size importance** | Provides framework for selecting appropriate temporal windows for taxi data |
| **Threshold strategies** | Survey covers adaptive threshold methods applicable to streaming |
| **Pattern types** | Defines point, contextual, and collective anomalies relevant to taxi data (e.g., unusual trips, rate surges) |
| **Not taxi-specific** | Survey focuses on univariate time-series; taxi data is inherently multivariate |

---

### Relevance to CA-DQStream

This batch of papers provides foundational methods and validation for the CA-DQStream approach:

| Paper | Key Contribution | CA-DQStream Application |
|-------|------------------|-------------------------|
| K-Means-IF | Geospatial anomaly detection, missing value handling | Detecting anomalous taxi coordinates, handling incomplete records |
| RIFIFI | Local density awareness, unified detection+clustering | Distinguishing anomalies in taxi-dense vs. sparse zones |
| IForestASD | Streaming IF with window-based adaptation | Core methodology for streaming taxi anomaly detection |
| Time-Series Survey | Method benchmarking, window selection | Framework for selecting temporal aggregation windows |

**Common Thread:** All papers emphasize that Isolation Forest variants are well-suited for anomaly detection in transportation/taxi data, with particular strength in handling geospatial features and varying data densities. The streaming adaptation (IForestASD) provides the most direct template for CA-DQStream's streaming architecture.

---

## 6b. MemStream Extensions & ADWIN Variants

This section surveys related work that extends MemStream's memory-augmented paradigm, proposes improved ADWIN-based concept drift detectors, and evaluates streaming anomaly detection methods on common benchmarks.

---

### 6b.1 ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection (2025)

**Paper**: [ADWIN-U: adaptive windowing for unsupervised drift detection on data streams](https://link.springer.com/article/10.1007/s10115-025-02523-1)
**Authors**: Daniel Nowak Assis, Vinicius M. A. Souza
**Venue**: Knowledge and Information Systems (KAIS), Vol. 67, pp. 10005–10034, November 2025
**Year**: 2025
**DOI**: [10.1007/s10115-025-02523-1](https://doi.org/10.1007/s10115-025-02523-1)

**Method Proposed**:
ADWIN-U extends the classical ADWIN (Adaptive Windowing) algorithm to operate in a fully unsupervised setting, removing the dependency on labeled data for drift monitoring. Traditional ADWIN relies on classifier accuracy metrics derived from labeled instances to detect distribution changes. ADWIN-U instead uses a novel unsupervised drift detection criterion based on statistical properties of the data stream itself, making it suitable for real-world scenarios where labeled data is costly or impractical. The authors also propose a new evaluation metric called **BAR** (Balanced Accuracy by the Amount of Requested Labeled Data), which explicitly rewards detectors that achieve high accuracy while minimizing reliance on labeled data.

**Key Results / Metrics**:
- ADWIN-U outperforms its supervised counterpart across multiple domains
- The BAR metric reveals that unsupervised ADWIN-U achieves comparable or better drift detection accuracy while requiring significantly fewer labeled instances
- Evaluated on synthetic and real-world streaming datasets covering diverse drift types (abrupt, gradual, incremental)

**How It Extends MemStream / ADWIN**:
ADWIN-U builds on the ADWIN framework (the same conceptual foundation used in MemStream for change detection). While MemStream uses a memory module to track normal behavior, ADWIN-U modifies the statistical windowing mechanism to detect drift without labels. This addresses a key practical limitation: in streaming quality monitoring, ground-truth labels (e.g., confirmed quality anomalies) are rarely available for model retraining triggers.

**Datasets Used**:
- Synthetic data streams with controlled drift patterns (abrupt, gradual, incremental)
- Real-world datasets from multiple application domains

---

### 6b.2 Adaptive-Delta ADWIN: Stability-Adaptive Framework for Intrusion Detection (2025)

**Paper**: [Adaptive-Delta ADWIN: A Framework for Stable and Sensitive Intrusion Detection in Streaming Networks](https://journal-isi.org/index.php/isi/article/view/1336)
**Author**: Rodney Buang Sebopelo (North-West University, South Africa)
**Venue**: Journal of Information Systems and Informatics, Vol. 7, No. 4, pp. 3711–3734, December 2025
**Year**: 2025
**DOI**: [10.63158/journalisi.v7i4.1336](https://doi.org/10.63158/journalisi.v7i4.1336)

**Method Proposed**:
Adaptive-Delta ADWIN introduces two online controllers that dynamically adjust the sensitivity parameter δ of the ADWIN algorithm in real time:
1. **Volatility Controller (VC)**: Monitors the rate of change in the data stream's statistical properties and increases δ (reduces sensitivity) during volatile periods to avoid spurious alarms
2. **AlertRate Controller (ARC)**: Monitors the alarm rate and fine-tunes δ to maintain a target false positive rate, ensuring stable operation

The framework is evaluated in the context of Network Intrusion Detection Systems (IDS) where concept drift between normal and malicious network traffic patterns is common.

**Key Results / Metrics**:
- Achieved accuracy of **0.93–0.95** on CICIDS2017 dataset using a multiclass ensemble of Hoeffding Adaptive Trees
- Outperformed fixed-δ baselines by up to **6.6%**
- False positive rate reduced by **50%**
- False negative rate reduced by **30%**
- Balances detection sensitivity with operational stability in real-time network conditions

**How It Extends MemStream / ADWIN**:
While MemStream uses a fixed update threshold β for its memory module, Adaptive-Delta ADWIN demonstrates the value of adaptive sensitivity control in streaming settings. For CA-DQStream, the insight is that quality score thresholds should adapt to the volatility of the underlying data stream — a static threshold will either be too noisy during stable periods or too slow to react during genuine quality shifts.

**Datasets Used**:
- **CICIDS2017**: Comprehensive network intrusion detection benchmark with realistic attack scenarios and traffic patterns

---

### 6b.3 Parallel ADWIN: Scalable Concept Drift Detection (EDBT 2018)

**Paper**: [Scalable Detection of Concept Drifts on Data Streams with Parallel Adaptive Windowing](https://tu-berlin-dima.github.io/parallel-ADWIN/)
**Authors**: Grulich, Sternke, Bifet, Rabl (TU Berlin, DIMA Group)
**Venue**: 21st International Conference on Extending Database Technology (EDBT), March 2018
**Year**: 2018
**GitHub**: [TU-Berlin-DIMA/parallel-ADWIN](https://github.com/TU-Berlin-DIMA/parallel-ADWIN)

**Method Proposed**:
Parallel ADWIN addresses the throughput bottleneck of the original ADWIN algorithm by introducing several parallelization strategies for multi-core and distributed environments. The authors implement multiple ADWIN variants:
- **SNAPSHOT**: Optimistic parallel ADWIN
- **HALFCUT**: Half-cut ADWIN variant
- Original and Serial baselines for comparison

The key insight is that the statistical comparison of sub-windows in ADWIN can be parallelized across independent windows, enabling scalable concept drift detection for high-velocity streams processing millions of tuples per second.

**Key Results / Metrics**:
- Throughput improvements of **two orders of magnitude** compared to serial ADWIN
- Enables scalable drift detection for high-velocity data streams
- Maintains detection accuracy comparable to the original ADWIN algorithm

**How It Extends MemStream / ADWIN**:
Parallel ADWIN demonstrates that streaming drift detection algorithms can be engineered for high throughput, a critical consideration for CA-DQStream when processing large-scale sensor or IoT data streams where quality monitoring must keep pace with data ingestion rates.

**Datasets Used**:
- Synthetic high-velocity data streams for benchmarking
- Evaluation of multiple ADWIN variants (SNAPSHOT, HALFCUT, ORIGINAL, SERIAL)

---

### 6b.4 ADWIN++: Optimizing ADWIN for Steady Streams (ACM SAC 2022)

**Paper**: [Optimizing ADWIN for Steady Streams](https://researchr.org/publication/MoharramAE22)
**Authors**: H. Moharram, A. Awad, P. M. El-Kafrawy
**Venue**: Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing (SAC), 2022
**Year**: 2022

**Method Proposed**:
ADWIN++ uses **adaptive bucket dropping** to control the window size in the ADWIN algorithm, specifically optimized for steady streams (data streams with no or negligible drift). Previous ADWIN variants focused on detecting drifts quickly but were memory-inefficient for steady-state operation. ADWIN++ introduces a mechanism to drop statistical evidence from the window when it is determined to be stale, reducing memory consumption without sacrificing drift detection capability.

**Key Results / Metrics**:
- **~80% memory savings** compared to the original ADWIN algorithm
- Faster drift detection while maintaining equivalent detection accuracy
- Evaluated on datasets covering different drift types: incremental, gradual, abrupt, and steady

**How It Extends MemStream / ADWIN**:
ADWIN++ addresses memory efficiency for the steady-state case. Since MemStream's memory module also grows with incoming data, ADWIN++'s bucket dropping mechanism provides a complementary strategy for bounding memory usage. For CA-DQStream, this suggests that quality reference data can be periodically pruned without losing the ability to detect genuine quality shifts.

**Datasets Used**:
- Synthetic datasets with various drift types (incremental, gradual, abrupt, steady)
- Real-life datasets for practical evaluation

---

### 6b.5 Revisiting Streaming Anomaly Detection: Benchmark and Evaluation (arXiv 2024)

**Paper**: [Revisiting Streaming Anomaly Detection: Benchmark and Evaluation](https://arxiv.org/html/2405.00704v2)
**arXiv ID**: 2405.00704
**Year**: 2024

**Method Proposed**:
This paper provides a comprehensive benchmark and evaluation of streaming anomaly detection methods, placing MemStream and related methods in the context of modern streaming ML evaluation practices. The authors identify common pitfalls in streaming anomaly detection benchmarks (e.g., incorrect handling of prequential evaluation, leakage in train/test splits, inconsistent scoring metrics) and propose rigorous evaluation protocols. The paper evaluates methods including river-based detectors and compares their performance, memory efficiency, and adaptation speed across datasets.

**Key Results / Metrics**:
- Identifies significant variability in reported metrics across existing benchmarks
- Proposes standardized evaluation protocols for streaming anomaly detection
- Compares methods on standard datasets including KDD-Cup, NSL-KDD, and ODDS datasets

**How It Extends MemStream / ADWIN**:
This benchmark study provides the evaluation framework within which MemStream variants (including River's MemStreamPCA) are assessed. For CA-DQStream, it establishes best practices for evaluation methodology — in particular, the importance of measuring detection delay, false positive rates, and adaptation time as separate metrics rather than aggregating them into a single AUC score.

**Datasets Used**:
- KDDCup99, NSL-KDD, UNSW-NB15
- ODDS benchmark datasets (Ionosphere, Cardio, Satellite, Satimage-2, Mammograph, Pima, ForestCover)
- Streaming-specific benchmarks with concept drift

---

### 6b.6 River Library: MemStream Integration (2025–2026)

**Paper**: [Proposed addition: MemStream anomaly detection for River · online-ml/river Discussion #1740](https://github.com/online-ml/river/discussions/1740)
**PR**: [Fix/memstreamriver Pull Request #1748](https://github.com/online-ml/river/pull/1748)
**Year**: 2025–2026

**Method Proposed**:
The River online machine learning library has been working on integrating MemStream into its anomaly detection module. The implementation includes:
- **MemStreamPCA**: A PCA-based variant compatible with River's numpy-based API
- **MemStreamAE**: A denoising autoencoder-based variant (proposed for deep-river)
- Configurable memory replacement strategies (FIFO, LRU, Random)
- Flexible scoring mechanisms and k-nearest neighbor weighting
- Grace period mechanism (default 5,000 samples) for encoder bootstrapping

**Key Features**:
- **Online Learning**: Processes data points one at a time without requiring labels
- **Concept Drift Adaptation**: Memory module evolves to handle distribution changes
- **Memory Management**: Configurable size (default 1,000 samples) and replacement policies
- **K-NN Scoring**: Uses k-nearest neighbors with exponential weighting for anomaly scores

**How It Extends MemStream / ADWIN**:
The River integration makes MemStream accessible to the broader streaming ML community and provides a standardized API for comparison. For CA-DQStream, River's MemStream implementation serves as a reference baseline — if CA-DQStream's quality detection can be expressed in River's `score_one` paradigm, it could be directly compared against MemStream and other detectors in the River benchmark suite.

**Datasets Used**:
- KDDCUP99, NSL-KDD, UNSW-NB15
- CICIDS-DoS, ODDS datasets

---

### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Adaptive sensitivity (ADWIN-U, Adaptive-Delta)** | CA-DQStream's quality thresholds should adapt to stream volatility — using ADWIN-based change detection on quality scores can automatically adjust sensitivity |
| **Memory optimization (ADWIN++)** | Provides methodology for bounding memory usage of quality reference data without losing detection capability; 80% memory savings applicable to quality profile storage |
| **Parallel ADWIN throughput** | CA-DQStream must keep pace with high-frequency streaming data; parallel ADWIN demonstrates 100× throughput gains achievable through parallelization |
| **River integration** | MemStreamPCA/MemStreamAE in River provide standardized baselines; CA-DQStream can be benchmarked against these using River's evaluation framework |
| **Benchmark protocols** | The arXiv 2024 benchmark study establishes rigorous evaluation methodology (detection delay, FPR, adaptation time as separate metrics) for CA-DQStream |
| **Unsupervised drift detection** | CA-DQStream operates without ground-truth quality labels; ADWIN-U's unsupervised drift detection is directly applicable to triggering quality model updates |
| **Grace period mechanism** | River's MemStream uses a grace period (5,000 samples) to bootstrap the encoder — CA-DQStream should similarly collect initial quality baselines before activating detection |
| **Memory replacement strategies** | FIFO, LRU, Random policies from River's MemStream can be evaluated for quality profile memory management |

---

### 6b.2 BibTeX Citations

```bibtex
@article{assis2025adwinu,
  title     = {ADWIN-U: adaptive windowing for unsupervised drift detection on data streams},
  author    = {Assis, Daniel Nowak and Souza, Vinicius M. A.},
  journal   = {Knowledge and Information Systems},
  volume    = {67},
  number    = {11},
  pages     = {10005--10034},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s10115-025-02523-1}
}

@article{sebopelo2025adaptivedelta,
  title     = {Adaptive-Delta ADWIN: A Framework for Stable and Sensitive Intrusion Detection in Streaming Networks},
  author    = {Sebopelo, Rodney Buang},
  journal   = {Journal of Information Systems and Informatics},
  volume    = {7},
  number    = {4},
  pages     = {3711--3734},
  year      = {2025},
  publisher = {journal-isi.org},
  doi       = {10.63158/journalisi.v7i4.1336}
}

@inproceedings{grulich2018parallel,
  title     = {Scalable Detection of Concept Drifts on Data Streams with Parallel Adaptive Windowing},
  author    = {Grulich, Lena and Sternke, Felix and Bifet, Albert and Rabl, Tilmann},
  booktitle = {Proceedings of the 21st International Conference on Extending Database Technology (EDBT)},
  pages     = {526--537},
  year      = {2018},
  publisher = {OpenProceedings}
}

@inproceedings{moharram2022adwinpp,
  title     = {Optimizing ADWIN for Steady Streams},
  author    = {Moharram, H. and Awad, A. and El-Kafrawy, P. M.},
  booktitle = {Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing (SAC)},
  pages     = {480--487},
  year      = {2022},
  publisher = {ACM},
  doi       = {10.1145/3477314.3507255}
}

@article{revisiting2024streaming,
  title     = {Revisiting Streaming Anomaly Detection: Benchmark and Evaluation},
  author    = {Huang, W. and Chen, Y. and Zhang, J. and others},
  journal   = {arXiv preprint arXiv:2405.00704},
  year      = {2024},
  eprint    = {2405.00704},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}

@misc{river2025memstream,
  title     = {Proposed addition: MemStream anomaly detection for River},
  author    = {{online-ml/river Contributors}},
  year      = {2025},
  howpublished = {\url{https://github.com/online-ml/river/discussions/1740}},
  note      = {GitHub Discussion #1740, accessed 2025}
}

## 9c. Demand Prediction & Drift-Aware Taxi ML (Batch 3)

### Paper 1: BRIGHT — Drift-Aware Demand Predictions for Taxi Networks

| Attribute | Details |
|-----------|---------|
| **Authors** | Amal Saadallah, Luís Moreira-Matias, Ricardo Sousa, Jihed Khiari, Erik Jenelius, João Gama |
| **Venue** | IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol. 32, No. 2, February 2020 |
| **DOI** | 10.1109/TKDE.2018.2883616 |

#### Method Overview

BRIGHT is a drift-aware supervised learning framework for taxi passenger demand prediction that handles concept drift explicitly. It operates as an ensemble of multiple time series forecasting methods with three key properties:

1. **Diversity of base learners**: Five different models with distinct memory sizes and feature spaces
2. **Drift-aware adaptation**: Explicit drift detection and informed adaptation mechanisms
3. **Two-stage ensemble**: Model selection via clustering + weighted averaging via loss

**Base Learners:**
- **TVPP** (Time-Varying Poisson Process): Simple probabilistic model with time-dependent service rates; handles incremental drift blindly
- **FF-TVPP** (Fading-Factor TVPP): Extends TVPP with fading factors for gradual drift adaptation
- **ARIMA**: Traditional autoregressive model with incremental parameter updates via delta rule
- **L1-VARX**: L1-regularized Vector Autoregressive model with exogenous variables (weather) for multivariate forecasting
- **Drift-Aware VAR**: Novel model with top-k neighborhood selection using Hoeffding inequality for drift detection

**Ensemble Mechanism:**
- Models are grouped into families via Gaussian Mixture Model clustering based on output variance
- Page-Hinkley (PH) test monitors residuals to detect drift and trigger model selection changes
- Hoeffding bound detects neighborhood structure changes in Drift-Aware VAR
- Final prediction: weighted average where weights are inversely proportional to recent loss

#### Key Results

**sMAPE Performance on Real-World Datasets:**

| Method | Porto (A) | Stockholm (B) | Shanghai (C) | Synthetic (X) |
|--------|-----------|--------------|--------------|---------------|
| FF-TVPP | 17.28% | 21.46% | 22.26% | 15.73% |
| ARIMA | 18.33% | 21.35% | 24.26% | 15.00% |
| L1-VARX | 18.36% | 18.93% | 22.09% | 11.26% |
| Drift-Aware VAR | 17.20% | 19.71% | 22.25% | 10.85% |
| SoA Ensemble | 16.49% | 19.51% | 21.11% | 14.68% |
| **BRIGHT** | **16.09%** | **18.56%** | **20.89%** | **11.88%** |

- **Peak Hour Performance**: During peak hours on synthetic data, BRIGHT achieves 15.93% sMAPE vs SoA Ensemble's 19.94% — a 25% improvement
- **Variance Reduction**: Up to 50% reduction in variance-type error while maintaining stable bias
- **Generalization**: 4% operational revenue improvement potential when deployed to taxi drivers

#### Datasets

1. **Porto, Portugal**: 438 taxis, July 2013–July 2014, ~1M trips
2. **Stockholm, Sweden**: 1.6K taxis, Jan 2014–Jan 2015, ~5.5M trips
3. **Shanghai, China**: 5.4K taxis, Feb 2006–Mar 2007, ~12.3M trips
4. **Semi-Synthetic (X)**: Tennessee Eastman Process-based generator with injected drifts

#### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Drift detection mechanisms** | Page-Hinkley test and Hoeffding bound provide templates for detecting quality concept drift in streaming taxi data |
| **Ensemble diversity** | Model family clustering could inspire diverse quality estimators in CA-DQStream |
| **Neighborhood-based forecasting** | Spatial neighborhood dynamics in taxi demand mirror quality signal dependencies |
| **Weighted loss averaging** | CA-DQStream could weight historical quality scores by recency |
| **Multi-city validation** | Porto/Stockholm/Shanghai setup provides model for CA-DQStream multi-region deployment |

---

### Paper 2: IForest-KMeans — Extending Isolation Forest for Anomaly Detection in Big Data

| Attribute | Details |
|-----------|---------|
| **Authors** | Md Tahmid Rahman Laskar, Jimmy Huang, Vladan Smetana, Chris Stewart, Kees Pouw, Aijun An, Stephen Chan, Lei Liu |
| **Venue** | arXiv:2104.13190v1, April 2021 |
| **Domain** | Network Intrusion Detection (Industrial Big Data) |

#### Method Overview

IForest-KMeans combines Isolation Forest with K-Means clustering to address two fundamental issues in Isolation Forest for big data scenarios:

1. **Threshold determination**: Isolation Forest requires a contamination ratio parameter and quantile calculation to convert anomaly scores to labels
2. **Parameter sensitivity**: Performance degrades with approximate quantile relative error settings

**Approach:**
1. Train Isolation Forest to generate anomaly scores for each data point
2. Use K-Means (K=2) to partition anomaly scores into clusters (normal/anomaly)
3. Cluster with fewer instances is designated as anomalous

**Apache Spark Implementation:**
- PySpark-based for distributed processing
- Handles 123M+ instances (1TB+ data) stored in Elasticsearch
- Real-time streaming evaluation via Spark Structured Streaming
- ~0.65 milliseconds per instance processing time

#### Key Results

**Industrial Dataset Performance (Cyberattack Detection):**

| Metric | IForest-KMeans | Isolation Forest (AQRE=0.5) | Isolation Forest (1% data) |
|--------|----------------|---------------------------|---------------------------|
| Port 3389 Attack Detection | 6,693/8,296 detected | 965 | 720 |
| Specific IP Attack (91% accuracy) | 13,513/14,917 | 0 detected | 0 detected |

**Academic Dataset AUC-ROC Comparison:**

| Dataset | IForest-KMeans | PySpark IForest | Original IForest |
|---------|----------------|-----------------|-----------------|
| Http (KDD) | 0.96 | 0.97 | 1.00 |
| ForestCover | 0.88 | 0.78 | 0.88 |
| Mulcross | 0.92 | 0.85 | 0.97 |
| Shuttle | 0.98 | 0.97 | 1.00 |
| Breastw | 0.98 | 0.67 | 0.99 |
| Annthyroid | 0.75 | 0.66 | 0.82 |

- Outperforms PySpark Isolation Forest on 10/12 datasets (statistically significant, p ≤ 0.05)
- Comparable to original single-machine IForest implementations

#### Datasets

1. **iSecurity Industrial Dataset**: 123M training instances (Jan 8-14, 2019), 34M test instances (Jan 15-16, 2019)
2. **12 Academic Benchmark Datasets**: Http, Smtp (KDD CUP 99), ForestCover, Shuttle, Mammography, Annthyroid, Satellite, Pima, Breastw, Arrhythmia, Ionosphere, Mulcross

#### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Isolation Forest foundation** | IForest-KMeans extends iForest which is directly relevant to CA-DQStream's quality scoring mechanism |
| **Big data streaming** | Real-time processing architecture (0.65ms/instance) provides deployment model for streaming quality monitoring |
| **Clustering-based scoring** | K-Means on anomaly scores provides alternative to threshold-based quality classification |
| **Spark integration** | Elasticsearch + Spark + Kibana stack is directly applicable to streaming quality dashboards |
| **Industrial deployment** | iSecurity use case mirrors production taxi data quality monitoring requirements |

---

### Paper 3: LaF-AD — Label-Free Anomaly Detection with Model Selection

| Attribute | Details |
|-----------|---------|
| **Authors** | Deokwoo Jung, Nandini Ramanan, Mehrnaz Amjadi, Sankeerth Rao Karingula, Jake Taylor, Claudionor Nunes Coelho Jr |
| **Venue** | arXiv:2106.07473v1, June 2021 |
| **Organization** | Palo Alto Networks, Advanced Applied AI Research |

#### Method Overview

LaF-AD (Label-free Anomaly Detection) addresses the challenge of model selection without labeled data by:

1. **Unsupervised ensemble learning** across multiple candidate parametric models
2. **Model variance metric** using bootstrapping to quantify anomaly probability sensitivity
3. **Collective decision** via model learners using variance-based weighting

**Boosted Embedding Model:**
- Combines gradient boosting with embeddings (DeepGB) to learn seasonality (daily, weekly, monthly)
- Captures categorical features via embeddings and residual modeling
- Handles ill-conditioned data with missing/corrupted samples

**Anomaly Scoring:**
- Embedding function f_emb learns normal patterns from unlabeled data
- Gaussian Mixture Model (GMM, k=2) maps anomaly distance to probability
- Ensemble weights derived from model variance: w_m = (1 - 4σ²_m) / Σ(1 - 4σ²_m)

#### Key Results

**NYC Taxi Dataset (NAB Benchmark):**

| Window Size | LaF-AD | Isolation Forest | KNN | AE-LSTM |
|------------|--------|----------------|-----|---------|
| 5 points | **0.82** | 0.44 | 0.82 | 0.53 |
| Variance | 0.0023 | 0.0041 | 0.0023 | 0.0042 |

- LaF-AD consistently outperforms Isolation Forest and AE-LSTM
- Achieves best performance with smallest model variance (most stable)
- Validates label-free model selection approach

#### Dataset

- **NYC Taxi**: 10,320 samples, July 2014–January 2015, 30-minute intervals, 5 labeled anomalies (holidays, weather events)

#### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Label-free model selection** | Directly addresses CA-DQStream challenge of quality assessment without ground truth labels |
| **Ensemble model diversity** | Bootstrapped variance metric provides methodology for quality model ensemble selection |
| **Seasonality learning** | Boosted embeddings capture taxi demand cycles relevant to quality pattern understanding |
| **Model variance as quality indicator** | Low variance correlates with reliable predictions — analogous to stable quality scores |
| **NAB taxi data** | NYC taxi dataset directly relevant to taxi anomaly detection benchmarking |

---

### Paper 4: Deep Isolation Forest (DIF)

| Attribute | Details |
|-----------|---------|
| **Authors** | Hongzuo Xu, Guansong Pang, Yijie Wang, Yongjun Wang |
| **Venue** | IEEE Transactions on Knowledge and Data Engineering (TKDE), arXiv:2206.06602v4 |
| **Code** | https://github.com/xuhongzuo/deep-iforest |

#### Method Overview

DIF addresses fundamental limitations of Isolation Forest by:

1. **Random Representation Ensemble**: Uses casually initialized neural networks (no training/optimization) to map original data into diverse representation spaces
2. **Non-linear Isolation**: Axis-parallel cuts on transformed spaces are equivalent to non-linear partitions in original space
3. **Algorithmic Bias Elimination**: Removes "ghost region" problem where iForest assigns unexpectedly low scores to artefact regions

**Two Key Innovations:**

**CERE (Computation-Efficient Representation Ensemble):**
- Vectorized computation via Hadamard product for efficient ensemble generation
- Linear time complexity O(N·D·r·t) — maintains iForest scalability
- ~3GB memory for 10,000 features with r=50, batch=64

**DEAS (Deviation-Enhanced Anomaly Scoring):**
- Combines path length with deviation degree from branching thresholds
- g(x|τ) = (1/|p(x|τ)|) Σ|x^(jk) - η_k| measures isolation difficulty
- Final score: F_DEAS = 2^(-E|p|/C(T)) × E[g(x|τ)]

#### Key Results

**Tabular Data AUC-ROC Performance:**

| Dataset | DIF | EIF | PID | LeSiNN | iForest |
|---------|-----|-----|-----|--------|---------|
| Analysis | **0.931** | 0.910 | 0.820 | 0.903 | 0.782 |
| Backdoor | **0.918** | 0.902 | 0.808 | 0.894 | 0.731 |
| DoS | **0.932** | 0.918 | 0.802 | 0.896 | 0.747 |
| Cover | **0.972** | 0.872 | 0.939 | 0.885 | 0.888 |
| Fraud | **0.953** | 0.950 | 0.950 | 0.952 | 0.950 |

- Significantly outperforms iForest and extensions (p < 0.01)
- 61% AUC-PR improvement over EIF on average
- Superior on high-dimensional data (R8, Analysis, Backdoor, DoS)

**Time Series Results (UCR Benchmark):**

| Dataset | DIF | EIF | iForest | TranAD |
|---------|-----|-----|---------|--------|
| Mars | 0.952 | **0.980** | 0.947 | 0.947 |
| ECG | **0.997** | 0.986 | 0.987 | 0.976 |
| ECG-w | **1.000** | 0.988 | 0.981 | 0.990 |

#### Datasets

- **10 Tabular**: Analysis, Backdoor, DoS, Exploits (UNSW-NB15), R8, Cover, Fraud, Pageblocks, Shuttle, Thrombin
- **4 Graph**: HSE, MMP, p53, PPAR (Tox21)
- **4 Time Series**: Mars, Gait, ECG, ECG-w (UCR benchmark)

#### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Non-linear isolation** | DIF's ability to detect hard anomalies in complex data spaces applicable to multivariate taxi quality anomalies |
| **Representation learning** | Random neural network transformations without training offers efficient feature engineering for quality scoring |
| **Scalability** | O(N·D) complexity maintains streaming feasibility for high-volume taxi data |
| **Ghost region elimination** | DIF's smooth score maps address false negatives — critical for quality assurance |
| **Multi-modal data** | DIF handles tabular, graph, and time series data — extensible to taxi GPS, network, and demand data |

---

### BibTeX Citations

```bibtex
@article{saadallah2020bright,
  title={BRIGHT—Drift-Aware Demand Predictions for Taxi Networks},
  author={Saadallah, Amal and Moreira-Matias, Lu{\'\i}s and Sousa, Ricardo and Khiari, Jihed and Jenelius, Erik and Gama, Jo{\~a}o},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={32},
  number={2},
  pages={234--245},
  year={2020},
  publisher={IEEE},
  doi={10.1109/TKDE.2018.2883616}
}

@article{laskar2021extending,
  title={Extending Isolation Forest for Anomaly Detection in Big Data via K-Means},
  author={Laskar, Md Tahmid Rahman and Huang, Jimmy and Smetana, Vladan and Stewart, Chris and Pouw, Kees and An, Aijun and Chan, Stephen and Liu, Lei},
  year={2021},
  eprint={2104.13190},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}

@article{jung2021lafad,
  title={Time Series Anomaly Detection with Label-free Model Selection},
  author={Jung, Deokwoo and Ramanan, Nandini and Amjadi, Mehrnaz and Karingula, Sankeerth Rao and Taylor, Jake and Coelho Jr, Claudionor Nunes},
  year={2021},
  eprint={2106.07473},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{xu2022deep,
  title={Deep Isolation Forest for Anomaly Detection},
  author={Xu, Hongzuo and Pang, Guansong and Wang, Yijie and Wang, Yongjun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  note={arXiv:2206.06602v4},
  doi={10.1109/TKDE.2022.3183023}
}
```

---

## 8. Citation-Ready References

This section compiles all full citations extracted from the research notes into three organized sub-sections for use in thesis chapters.

---

### 8.1 Primary Method References (Background & Related Work)

#### 8.1.1 Streaming Anomaly Detection Methods

---

**MemStream: Memory-Based Streaming Anomaly Detection**

**APA Citation:**
Bhatia, S., Jain, A., Srivastava, S., Kawaguchi, K., & Hooi, B. (2022). MemStream: Memory-Based Streaming Anomaly Detection. In *Proceedings of the ACM Web Conference 2022 (WWW '22)* (pp. 1–12). ACM. https://doi.org/10.1145/3485447.3512221

**BibTeX Entry:**
```bibtex
@inproceedings{bhatia2022memstream,
  title     = {MemStream: Memory-Based Streaming Anomaly Detection},
  author    = {Bhatia, Siddharth and Jain, Arjit and Srivastava, Shivin
               and Kawaguchi, Kenji and Hooi, Bryan},
  booktitle = {Proceedings of the ACM Web Conference 2022 (WWW '22)},
  pages     = {1--12},
  year      = {2022},
  publisher = {ACM},
  doi       = {10.1145/3485447.3512221}
}
```

**URL:** https://github.com/Stream-AD/MemStream | Integrated into River library (online-ml/river, PR #1748)

---

**SCAR Benchmark: Revisiting Streaming Anomaly Detection**

**APA Citation:**
Ma, Y., et al. (2025). Revisiting streaming anomaly detection: Benchmark and evaluation. *Artificial Intelligence Review*, 58, 8. Springer. https://doi.org/10.1007/s10462-024-10995-w

**BibTeX Entry:**
```bibtex
@article{scar2025,
  title   = {Revisiting streaming anomaly detection: Benchmark and evaluation},
  author  = {Ma, Y. and others},
  journal = {Artificial Intelligence Review},
  volume  = {58},
  number  = {8},
  year    = {2025},
  publisher = {Springer},
  doi     = {10.1007/s10462-024-10995-w}
}
```

**URL:** Evaluates 9 streaming algorithms + 4 adapted static baselines (iForest, LOF, iNNE, IDK) across 76 synthetic and 74 real-world datasets using per-segment AUC-ROC.

---

**ADBench: Anomaly Detection Benchmark**

**APA Citation:**
Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 35, pp. 15542–15563).

**BibTeX Entry:**
```bibtex
@inproceedings{adbench2022,
  title     = {ADBench: Anomaly Detection Benchmark},
  author    = {Han, S. and Hu, X. and Huang, H. and Jiang, M. and Zhao, Y.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  pages     = {15542--15563},
  year      = {2022}
}
```

**URL:** https://github.com/Minqi824/ADBench | 30 algorithms × 57 datasets (47 existing + 10 new from CV/NLP)

---

**Isolation Forest (Foundational Paper)**

**APA Citation:**
Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. In *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM)* (pp. 413–422). IEEE. https://doi.org/10.1109/ICDM.2008.17

**BibTeX Entry:**
```bibtex
@inproceedings{liu2008iforest,
  title     = {Isolation Forest},
  author    = {Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle = {Proceedings of the 2008 Eighth IEEE International Conference
               on Data Mining (ICDM)},
  pages     = {413--422},
  year      = {2008},
  publisher = {IEEE},
  doi       = {10.1109/ICDM.2008.17}
}
```

---

**Isolation Forest: An In-Depth Study and Improvement**

**APA Citation:**
Chabchoub, Y., Togbe, M. U., Boly, A., & Chiky, R. (2022). An in-depth study and improvement of Isolation Forest. *IEEE Access*, 10, 10219–10247. https://doi.org/10.1109/ACCESS.2022.3144425

---

**RRCF: Robust Random Cut Forest**

**Note:** RRCF was developed at Amazon for anomaly detection on streaming data. Used as a baseline in the NYC taxi anomaly detection literature (K-Lab, 2018).

**URL:** https://github.com/klabum/rrcf

---

#### 8.1.2 NYC Taxi Anomaly Detection Papers

---

**Kuo et al. (CIKM 2018): Detecting Outliers in Data with Correlated Measures**

**APA Citation:**
Kuo, Y.-H., Li, Z., & Kifer, D. (2018). Detecting Outliers in Data with Correlated Measures. In *Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM '18)* (pp. 547–556). ACM. https://doi.org/10.1145/3269206.3271798

**BibTeX Entry:**
```bibtex
@inproceedings{kuo2018cikm,
  title     = {Detecting Outliers in Data with Correlated Measures},
  author    = {Kuo, Yu-Hsuan and Li, Zhenhui and Kifer, Daniel},
  booktitle = {Proceedings of the 27th ACM International Conference on
               Information and Knowledge Management (CIKM '18)},
  pages     = {547--556},
  year      = {2018},
  publisher = {ACM},
  doi       = {10.1145/3269206.3271798}
}
```

---

**Baier et al. (arXiv 2020): Switching Scheme for Incremental Concept Drift**

**APA Citation:**
Baier, L., Kellner, V., Kühl, N., & Satzger, G. (2020). Switching Scheme: A Novel Approach for Handling Incremental Concept Drift in Real-World Data Sets. *arXiv preprint* arXiv:2011.02738. Karlsruhe Institute of Technology.

**BibTeX Entry:**
```bibtex
@article{baier2020arxiv,
  title   = {Switching Scheme: A Novel Approach for Handling Incremental
             Concept Drift in Real-World Data Sets},
  author  = {Baier, Lucas and Kellner, Vincent and Kühl, Niklas
             and Satzger, Gerhard},
  journal = {arXiv preprint arXiv:2011.02738},
  year    = {2020}
}
```

**URL:** https://arxiv.org/abs/2011.02738 | Uses NYC Taxi demand data; compares ADWIN, STEPD, HDDDM for drift detection

---

**Rahman et al. (AusDM 2025): DriftSense**

**APA Citation:**
Rahman, M. M., Mamun, Q., Bewong, M., & Islam, M. Z. (2026). DriftSense: Adaptive Drift Detection with Incremental Hoeffding Trees for Real-Time Spatial Crowdsourcing. In *Q.V. Nguyen et al. (Eds.), Data Science and Machine Learning (AusDM 2025)*. Communications in Computer and Information Science, vol 2765. Springer. https://doi.org/10.1007/978-981-95-6786-7_7

**BibTeX Entry:**
```bibtex
@inproceedings{rahman2026ausdm,
  title     = {DriftSense: Adaptive Drift Detection with Incremental Hoeffding
               Trees for Real-Time Spatial Crowdsourcing},
  author    = {Rahman, Md Mujibur and Mamun, Quazi and Bewong, Michael
               and Islam, Md Zahidul},
  booktitle = {Data Science and Machine Learning (AusDM 2025)},
  editor    = {Nguyen, Q.V. and others},
  series    = {Communications in Computer and Information Science},
  volume    = {2765},
  year      = {2025/2026},
  publisher = {Springer},
  doi       = {10.1007/978-981-95-6786-7_7}
}
```

**URL:** Evaluated on NYC Taxi dataset with injected abrupt, gradual, and mixed drift types; achieves 25% higher detection accuracy than baselines

---

**Malhotra et al. (2016): LSTM-based Encoder-Decoder for Anomaly Detection**

**APA Citation:**
Malhotra, P., et al. (2016). LSTM-based Encoder-Decoder for Multi-site Anthropogenic Damage Detection. *arXiv preprint*. (Foundational LSTM encoder-decoder for time series anomaly detection)

---

#### 8.1.3 Context-Aware / Spatio-Temporal Methods

---

**BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection (GeoAnomalies 2025)**

**APA Citation:**
Xie, J., Kim, J., Chiang, Y.-Y., Zhao, L., & Shafique, K. (2025). BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection for Human Mobility Data. In *Proceedings of the 2nd ACM SIGSPATIAL International Workshop on Geospatial Anomaly Detection (GeoAnomalies '25)*. ACM. https://doi.org/10.1145/3764914.3770888

**BibTeX Entry:**
```bibtex
@inproceedings{bestad2025,
  title     = {BeSTAD: Behavior-Aware Spatio-Temporal Anomaly Detection
               for Human Mobility Data},
  author    = {Xie, Junyi and Kim, Jina and Chiang, Yao-Yi
               and Zhao, Lingyi and Shafique, Khurram},
  booktitle = {Proceedings of the 2nd ACM SIGSPATIAL International Workshop
               on Geospatial Anomaly Detection (GeoAnomalies '25)},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3764914.3770888}
}
```

---

**ICAD: Interpretable Component-wise Anomaly Detection (SIGSPATIAL 2025)**

**APA Citation:**
Siampou, M. D., et al. (2025). ICAD: A Self-Supervised Autoregressive Approach for Multi-Context Anomaly Detection in Human Mobility Data. In *Proceedings of the ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM SIGSPATIAL 2025)* (pp. 595–606). ACM. https://doi.org/10.1145/3748636.3762774

**BibTeX Entry:**
```bibtex
@inproceedings{icad2025,
  title     = {ICAD: A Self-Supervised Autoregressive Approach for
               Multi-Context Anomaly Detection in Human Mobility Data},
  author    = {Siampou, Maria Despoina and others},
  booktitle = {Proceedings of the ACM SIGSPATIAL International Conference
               on Advances in Geographic Information Systems
               (ACM SIGSPATIAL 2025)},
  pages     = {595--606},
  year      = {2025},
  publisher = {ACM},
  doi       = {10.1145/3748636.3762774},
  pmcid     = {PMC13075516}
}
```

**URL:** https://github.com/USC-InfoLab/ICAD

---

**TAPS: Real-Time Taxi Spatial Anomaly Detection (Travel Behaviour & Society 2021)**

**APA Citation:**
Chen, B., et al. (2021). Real-time taxi spatial anomaly detection based on vehicle trajectory prediction. *Travel Behaviour and Society*, 34, 100698. https://doi.org/10.1016/j.tbs.2021.100698

**BibTeX Entry:**
```bibtex
@article{taps2021,
  title   = {Real-time taxi spatial anomaly detection based on
             vehicle trajectory prediction},
  author  = {Chen, Bingkun and others},
  journal = {Travel Behaviour and Society},
  volume  = {34},
  pages   = {100698},
  year    = {2021},
  doi     = {10.1016/j.tbs.2021.100698}
}
```

---

**MTRI: Multi-Scale Temporal Model for Vehicle Trajectory Anomaly Detection (Sustainability 2026)**

**APA Citation:**
Chen, J., Chen, H., & Lu, H. (2026). Enhancing Road Safety and Sustainability: A Multi-Scale Temporal Model for Vehicle Trajectory Anomaly Detection in Road Network Interactions. *Sustainability*, 18(2), 597. https://doi.org/10.3390/su18020597

**BibTeX Entry:**
```bibtex
@article{mtri2026,
  title   = {Enhancing Road Safety and Sustainability: A Multi-Scale Temporal
             Model for Vehicle Trajectory Anomaly Detection in Road
             Network Interactions},
  author  = {Chen, Juan and Chen, Haoran and Lu, Hongyu},
  journal = {Sustainability},
  volume  = {18},
  number  = {2},
  pages   = {597},
  year    = {2026},
  doi     = {10.3390/su18020597}
}
```

**URL:** Evaluated on Porto, Portugal real-world taxi/GPS trajectories; AUC-ROC > 0.85

---

**ReAD: Regional Anomaly Detection via Dynamic Partition (arXiv 2020)**

**APA Citation:**
Luo, H., Meng, C., Wu, B., Zhang, J., Li, T., & Zheng, Y. (2020). ReAD: A Regional Anomaly Detection Framework Based on Dynamic Partition. *arXiv preprint* arXiv:2007.06794. https://ar5iv.labs.arxiv.org/html/2007.06794

**BibTeX Entry:**
```bibtex
@article{read2020,
  title   = {ReAD: A Regional Anomaly Detection Framework Based
             on Dynamic Partition},
  author  = {Luo, Huaishao and Meng, Chuishi and Wu, Bowen
             and Zhang, Junbo and Li, Tianrui and Zheng, Yu},
  journal = {arXiv preprint},
  year    = {2020},
  note    = {arXiv:2007.06794},
  url     = {https://ar5iv.labs.arxiv.org/html/2007.06794}
}
```

---

#### 8.1.4 Concept Drift Detection Methods

---

**DTD: Autonomous Dynamic Threshold Determination for Concept Drift Detection (AAAI 2026)**

**APA Citation:**
Lu, P., Lu, J., Liu, A., Yu, E., & Zhang, G. (2026). Autonomous Concept Drift Threshold Determination. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2026)*. https://arxiv.org/html/2511.09953v1

**BibTeX Entry:**
```bibtex
@inproceedings{dtd2026,
  title     = {Autonomous Concept Drift Threshold Determination},
  author    = {Lu, Pengqian and Lu, Jie and Liu, Anjin
               and Yu, En and Zhang, Guangquan},
  booktitle = {Proceedings of the AAAI Conference on Artificial
               Intelligence (AAAI 2026)},
  year      = {2026},
  note      = {arXiv:2511.09953},
  url       = {https://arxiv.org/html/2511.09953v1}
}
```

**URL:** https://arxiv.org/abs/2511.09953 | Proves that no single fixed threshold can be universally optimal; DTD-enhanced HDDM-W achieves 58.31% accuracy vs. 48.64% with fixed threshold

---

**ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection (KAIS 2025)**

**APA Citation:**
Assis, D. N., & Souza, V. M. A. (2025). ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams. *Knowledge and Information Systems*, 67, 10005–10034. https://doi.org/10.1007/s10115-025-02523-1

**BibTeX Entry:**
```bibtex
@article{adwinu2025,
  title   = {ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection
             on Data Streams},
  author  = {Assis, D. N. and Souza, V. M. A.},
  journal = {Knowledge and Information Systems},
  volume  = {67},
  pages   = {10005--10034},
  year    = {2025},
  doi     = {10.1007/s10115-025-02523-1}
}
```

---

#### 8.1.5 Survey Papers

---

**Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey (arXiv 2025)**

**APA Citation:**
Huang, H., Wang, P., Pei, J., Wang, J., Alexanian, S., & Niyato, D. (2025). Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey. *arXiv preprint* arXiv:2503.13195. https://arxiv.org/abs/2503.13195

**BibTeX Entry:**
```bibtex
@article{survey2025,
  title   = {Deep Learning Advancements in Anomaly Detection:
             A Comprehensive Survey},
  author  = {Huang, Haoqi and Wang, Ping and Pei, Jianhua
             and Wang, Jiacheng and Alexanian, Shahen and Niyato, Dusit},
  journal = {arXiv preprint},
  year    = {2025},
  note    = {arXiv:2503.13195},
  url     = {https://arxiv.org/abs/2503.13195}
}
```

**Scope:** 180+ recent studies (2019–2024); covers supervised, semi-supervised, and unsupervised deep AD; reconstruction-based, prediction-based, and hybrid paradigms

---

**Self-Supervised Anomaly Detection: A Survey and Outlook (Neural Networks 2025)**

**APA Citation:**
Hojjati, H., Ho, T. K. K., & Armanfard, N. (2025). Self-Supervised Anomaly Detection: A Survey and Outlook. *Neural Networks*, 170, 1083–1096. https://doi.org/10.1016/j.neunet.2023.10.034

---

**Streaming Anomaly Detection: A Comparison and Evaluation Study (Expert Systems with Applications 2023)**

**APA Citation:**
Iglesias Vázquez, F., et al. (2023). Anomaly detection in streaming data: A comparison and evaluation study. *Expert Systems with Applications*, 213, Part C, 119296. https://doi.org/10.1016/j.eswa.2022.119296

---

**NAB: Numenta Anomaly Benchmark**

**Note:** Provides labeled NYC Taxi 30-minute aggregated passenger count data with 5 ground-truth anomalies: NYC Marathon, Thanksgiving, Christmas Day, New Year's Day, and January 2015 Blizzard.

**URL:** https://github.com/numenta/NAB | Data available at: https://github.com/numenta/NAB/tree/master/data/realKnownCause

---

### 8.2 Industry/Engineering References (Motivation & Related Work)

#### 8.2.1 Grab Engineering

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **Signals Marketplace (Data Mesh at Grab)** | https://engineering.grab.com/signals-market-place | Data mesh architecture; data certification; SLA guarantees; 75% of queries hitting certified assets; 58% reduction in P80 datasets |
| **Data First, SLA Always (Trailblazer CDC Pipeline)** | https://engineering.grab.com/data-first-sla-always | CDC pipeline via Debezium + Kafka + Spark Structured Streaming; DASH hourly discrepancy checks; petabyte-scale data lake; 60 Spark streaming jobs |
| **Data Observability (GrabDefence Risk Systems)** | https://engineering.grab.com/data-observability | Real-time (not batch) alerting via Flink SQL; 5-minute tumbling windows; Datadog anomaly detection; alerts reduced from days/weeks to same day/hour |
| **Real-time Data Quality Monitoring (Coban Platform)** | https://engineering.grab.com/real-time-data-quality-monitoring | FlinkSQL-based test runner for 100+ Kafka topics; syntactic + semantic error detection; LLM-assisted semantic rule recommendation; Genchi + Slack alerting |
| **Real-time Data Ingestion Architecture** | https://engineering.grab.com/real-time-data-ingestion | Debezium CDC; DynamoDB streams; Protobuf-encoded Kafka messages; Golang consumer writing to S3; 90% database read reduction |
| **Rethinking Stream Processing: Data Exploration** | https://engineering.grab.com/rethinking-streaming-processing-data-exploration | Apache Zeppelin + Flink interpreter; DDL derivation from Protobuf schemas; Strimzi + OPA security |
| **The Complete Stream Processing Journey on FlinkSQL** | https://engineering.grab.com/the-complete-stream-processing-journey-on-flinksql | 3-layer FlinkSQL gateway (Compute/Integration/Query); cold start 5min→1min; full pipeline deployment <10min |
| **AutoMQ Migration for Kafka Infrastructure** | https://www.automq.com/blog/how-grab-uses-automq-solve-kafka-challenges | AutoMQ shared-storage Kafka; 3x throughput; 3x cost efficiency; partition reassignment 6hrs→<1min |

---

#### 8.2.2 Confluent Blog Posts

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **Preventing and Fixing Bad Data in Event Streams, Part 1** | https://www.confluent.io/blog/shift-left-bad-data-in-event-streams-part-1/ | Eight types of bad data; shift-left prevention; three-tier mitigation hierarchy (prevention → event design → rewind/rebuild/retry); CEL-based data contracts; DLQ caution |
| **Making Data Quality Scalable With Real-Time Streaming Architectures** | https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/ | Shift-left validation; two-layer continuous quality framework (validation + monitoring); six-step real-time validation pipeline; batch validation limitations; Siemens Healthineers 8M messages/day case study |
| **Best Practices for Kafka Connect Data Transformation & Schema Management** | https://www.confluent.io/blog/kafka-connect-data-transformation-schema/ | Schema format selection (Avro/Protobuf/JSON Schema); BACKWARD/FORWARD/FULL compatibility; SMTs; exactly-once semantics; two-phase commit; PII handling |

---

#### 8.2.3 Uber Engineering

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **The Billion Data Point Challenge** | https://www.uber.com/ie/en/blog/billion-data-point-challenge/ | M3 query engine: 2,500 queries/sec; 8.5B data points/sec; 3.5–35 Gbps; LTTB downsampling preserving outliers; active-active multi-datacenter deployment; anomaly detection + resource estimation on M3 platform |

---

#### 8.2.4 Ververica / Fintech

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **Fintech Monitoring with Apache Flink** | https://www.ververica.com/banking/fintech-monitoring | VERA engine: 6.9B records/sec throughput; sub-10ms latency; exactly-once guarantees; 40% lower TCO vs. self-managed Flink; compliance monitoring (MiFID II, KYC); $4.7B regulatory fines in 2025 alone |
| **What is Apache Flink** | https://www.ververica.com/ecosystem-introduction/what-is-apache-flink | Apache Flink as the stream processing substrate; Alibaba: 1 trillion events/day, 470M transactions/sec at peak |

---

#### 8.2.5 IBM / Databricks / Precisely Data Quality

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **IBM Data Quality** | https://www.ibm.com/think/topics/data-quality | Seven dimensions of data quality; $12.9M average annual cost of poor data quality; "Garbage in, garbage out" for ML; data governance |
| **Databricks Data Quality** | https://www.databricks.com/blog/what-is-data-quality | Six dimensions; "Seven Cs of Data Quality" framework (Collect → Catalog); nearly $13M annual cost of poor data quality |
| **Precisely: Big Data Quality** | https://www.precisely.com/blog/data-quality/big-data-quality-mastering-data-quality-in-the-age-of-big-data/ | Data in motion vulnerability; 5-step framework (Discover → Define → Design → Deploy → Monitor); big data challenges (speed, variation, enormity) |

---

#### 8.2.6 NYC TLC Official Data

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **NYC TLC Trip Record Data** | https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page | Official source for Yellow/Green taxis, FHV, and High Volume FHV datasets in PARQUET format; ~2 month publication delay; congestion fee column added 2025+ |

---

#### 8.2.7 GitHub / Open Source Projects

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **MemStream (Stream-AD)** | https://github.com/Stream-AD/MemStream | Official MemStream implementation; Apache 2.0 license; Python + PyTorch; KDDCUP99, NSL-KDD, UNSW-NB15, ODDS datasets |
| **River (online-ml)** | https://github.com/online-ml/river | Online machine learning library; MemStream integrated via PR #1748; streaming anomaly detection algorithms |
| **NYC Taxi Streaming (techatspree)** | https://github.com/techatspree/nyc-taxi-streaming | Kappa architecture: Python producer → Kafka → Flink → Elasticsearch + Kibana |
| **RRCF Library (K-Lab)** | https://github.com/klabum/rrcf | RRCF Python implementation; NYC taxi anomaly detection example with shingling; 48-step rolling windows |
| **NAB (Numenta)** | https://github.com/numenta/NAB | Numenta Anomaly Benchmark; labeled NYC taxi 30-minute aggregation with 5 ground-truth anomalies |

---

#### 8.2.8 Additional Technical References

| Source | URL | What It Supports |
|--------|-----|-----------------|
| **Isolation Forest NYC Taxi (SoftwareMill)** | https://softwaremill.com/isolation-forest-anomaly-detection-with-spark-and-nyc-taxi-data-stream/ | Spark Structured Streaming + Isolation Forest on NYC taxi; feature engineering (distance, meter increment, travel time); 4.9x fare anomaly detection |
| **Lyft Streaming Platform (Alibaba Cloud)** | https://www.alibabacloud.com/blog/lyfts-large-scale-flink-based-near-real-time-data-analytics-platform_596674 | Lyft's production Flink platform; minute-level streaming windows; ML-based feature engineering; cross-team Flink SQL access |
| **NYC Taxi Streaming Pipeline (Hamza Paracha)** | https://medium.com/@hamzaparacha098/phase-1-part-2-building-a-real-time-streaming-pipeline-with-kafka-spark-python-for-nyc-63f2471dedbd | Phase-by-phase Kafka + Spark Structured Streaming pipeline for NYC taxi data; Python producer simulation |
| **Drift with Evidently/MLflow (Coditation)** | https://www.coditation.com/blog/how-to-detect-drift-with-evidently-and-mlflow | Evidently AI integration with MLflow; 1000-event windows with 50% overlap; KS test p-value < 0.05 drift threshold |
| **Model Drift in Streaming (Conduktor)** | https://www.conduktor.io/glossary/model-drift-in-streaming | Schema contracts for drift prevention; topic-level governance (Gold/Bronze tiers) |
| **Striim: Data Quality for AI Analytics** | https://www.striim.com/blog/data-quality-availability-ai-analytics/ | Streaming data quality challenges; automated validation; real-time cleansing; lineage tracking |
| **Actian: Streaming Data Pipelines** | https://www.actian.com/streaming-data-pipelines/ | Lambda/Kappa architectures; decentralized data management; CI/CD for data pipelines |

---

### 8.3 Baseline Methods Summary Table

| Method | Paper/Source | Venue | Key Metric | CA-DQStream Use |
|--------|-------------|-------|------------|-----------------|
| **Isolation Forest (iForest)** | Liu et al. (2008) | ICDM 2008 | AUC-ROC, linear O(tψ log ψ) complexity | Primary baseline; fast, global anomaly detection; struggles with local anomalies |
| **MemStream** | Bhatia et al. (2022) | WWW 2022 | AUC-ROC (0.980 on KDD-CUP99) | Memory-based reference; FIFO adaptation; addresses concept drift; DAE + memory module |
| **RRCF** (Robust Random Cut Forest) | Amazon / K-Lab (2019) | — | CoDisp score; rolling window shingling | Streaming baseline; collaborative dispersion; competitive with iForest on NYC taxi |
| **LOF** (Local Outlier Factor) | Breunig et al. (2000) | SIGMOD 2000 | LOF score; density-based | Local anomaly detection baseline; O(n²) complexity limits on large streams |
| **HBOS** (Histogram-Based Outlier Score) | Goldstein & Dengel (2012) | — | AUC-ROC; execution time | Fast statistical baseline; treats features independently; good for real-time first-pass |
| **COPOD** (Copula-Based) | Li et al. (2020) | ICDM 2020 | AUC-ROC | Strong on correlated features; fast; handles mixed-type data |
| **ECOD** (Empirical CDF) | Li et al. (2021) | KDD 2021 | AUC-ROC | Fast; good for mixed-type; top performer in ADBench for speed-accuracy trade-off |
| **Denoising Autoencoder (DAE)** | MemStream (2022) | WWW 2022 | Reconstruction error | Feature extraction in MemStream; learns normal data representations |
| **LSTM Autoencoder** | Malhotra et al. (2016); GitHub anindya-saha | — | Reconstruction error; per-sequence scoring | Captures temporal dependencies in taxi demand; detects periodic patterns |
| **MTRI** (Multi-scale Temporal) | Chen et al. (2026) | Sustainability 2026 | AUC-ROC > 0.85 (Porto dataset) | Multi-scale temporal features; heterogeneous road graph; contrastive learning augmentation |
| **ARCUS** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Anomaly score distribution comparison for drift-triggered model updates |
| **iForestASD** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Conditional update on outlier rate change; built-in concept drift detection |
| **LODA** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Streaming density estimation; competitive but struggles on image data |
| **xStream** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Ensemble of random subspaces; streams ranking; handles complex data types |
| **MStream** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Memory-based streaming; predecessor to MemStream |
| **STORM** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Self-tuning online random forest; adaptive to concept drift |
| **HS-Trees** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Hoeffding tree ensemble for streaming; concept drift adaptation |
| **RS-Hash** | SCAR Benchmark (Ma et al., 2025) | AI Review 2025 | Per-segment AUC-ROC | Random hashing for streaming anomaly detection |

---

### 8.4 Suggested Research Gaps

Based on the comprehensive literature review across streaming anomaly detection, concept drift, context-aware systems, and industry data quality practices, the following research gaps are identified that CA-DQStream addresses:

1. **Lack of Context-Aware Data Quality for Streaming Pipelines:** Existing streaming data quality tools (e.g., Confluent data contracts, Grab Coban, Datadog anomaly detection) focus primarily on schema validation, rule-based checks, and generic statistical anomaly detection. None incorporate multi-dimensional contextual signals (geographic location, time-of-day, day-of-week, zone type) into real-time quality scoring. CA-DQStream's 5W1H contextual framework (Who, What, When, Where, Why, How) directly fills this gap by computing quality scores relative to learned contextual profiles rather than global thresholds.

2. **Memory-Based Methods Lack Geographic and Temporal Context for Urban Mobility Data:** MemStream and ADWIN-U maintain memory of historical data distributions but do not differentiate quality expectations across geographic zones or temporal periods. BeSTAD (Xie et al., 2025), ReAD (Luo et al., 2020), and DriftSense (Rahman et al., 2026) demonstrate that spatial and temporal context significantly improves anomaly detection in urban mobility data — yet no streaming quality system applies these insights to data quality monitoring. CA-DQStream introduces zone-aware and periodicity-aware quality profiles.

3. **Concept Drift Handling Lacks Automated Strategy Selection for Quality Signals:** The literature (Baier et al., 2020; ADWIN-U, 2025; DTD, 2026) demonstrates that drift detection thresholds and update strategies significantly impact performance, yet existing streaming DQ systems apply fixed adaptation strategies. DTD (Lu et al., AAAI 2026) formally proves that no single fixed threshold is universally optimal. CA-DQStream's adaptive threshold mechanism, informed by DTD's theoretical framework, enables automated strategy selection based on observed quality score behavior.

4. **Dual-Branch Rule-Based + ML-Based Quality Assessment is Underexplored:** Industry tools (Confluent, Grab, IBM) rely on rule-based validation; academic streaming anomaly detectors (MemStream, RRCF, LOF) rely on ML-based scoring. The synthesis of both branches — schema/semantic rules for structural quality + ML scoring for behavioral anomalies — within a unified streaming pipeline is not well studied. CA-DQStream's dual-branch architecture (rule evaluation engine + ML quality scorer) addresses this underexplored combination.

5. **Adaptive Per-Zone Quality Thresholds Are Not Well Studied for Streaming Data:** ReAD (Luo et al., 2020) demonstrates that dynamic, region-specific anomaly thresholds outperform fixed global thresholds in spatial anomaly detection. DTD (Lu et al., 2026) provides formal justification for dynamic thresholds in concept drift. No existing work applies adaptive per-zone thresholds to streaming data quality evaluation. CA-DQStream introduces geographic zone-stratified quality thresholds that adapt based on local data distribution characteristics.

6. **Real-Time Meta-Aggregation for Drift Monitoring Across Quality Dimensions is Underexplored:** Evidently AI, WhyLabs, and NannyML monitor ML model drift but do not monitor *data quality score drift* across multiple dimensions simultaneously. ADWIN-U (Assis & Souza, KAIS 2025) proves the value of unsupervised drift detection without ground truth labels. SCAR (Ma et al., 2025) shows that per-segment evaluation avoids mixing incompatible anomaly scores. CA-DQStream introduces a meta-aggregation layer that monitors drift in quality score distributions (completeness, timeliness, consistency, validity) across reference/current windows, enabling early warning of systemic quality degradation before individual dimension thresholds are breached.

---

## 12. Critical Comparison: Our Method vs. State-of-the-Art

This section critically evaluates CA-DQStream against existing approaches in the literature, honestly acknowledging where existing methods excel and where they fall short. The goal is not to inflate CA-DQStream's contributions but to precisely identify the specific gaps it fills in the research landscape.

### 12.1 What Existing Methods Do Well

The literature demonstrates several clear strengths across existing approaches:

**Transformer-based models (TFT/TST) show strong performance on temporal dependencies.** Dewi et al. (2025, JADS) report that the Temporal Fusion Transformer (TFT) achieves F1=0.92 and PR-AUC=0.71 on NYC taxi anomaly detection, outperforming LSTM Autoencoders (F1=0.85, PR-AUC=0.54) by ~8.24% in F1-score [Dewi et al., 2025, Journal of Applied Data Sciences]. The TST model processes data 30% faster than LSTM variants due to parallel processing, and TFT achieves 25% faster inference. These models excel at capturing long-range dependencies through self-attention mechanisms and handle multivariate temporal features well.

**Temporal Convolutional Autoencoders (TCN-AE) demonstrate superior robustness on complex industrial data.** Krasnikov et al. (2025) show TCN-AE achieving F1=0.991 with MSE=0.22 on process-complex industrial time series, significantly outperforming LSTM-AE (F1=0.853, MSE=1.23) and GRU-AE (F1=0.918, MSE=0.84) [Krasnikov et al., 2025, arXiv:2604.13928]. The key finding is that "architectural alignment with temporal structure is more critical than model complexity."

**Graph Neural Network Autoencoders capture spatial dependencies effectively.** The FT-AED benchmark (Coursey et al., 2024, NeurIPS) demonstrates that GCN-based autoencoders (AUC=0.70) outperform temporal-only models (Transformer AE: AUC=0.60) for freeway traffic anomaly detection [Coursey et al., 2024, NeurIPS Datasets & Benchmarks]. The GCN detects crashes 10.20 minutes before official reporting with 25% miss rate.

**Isolation Forest variants provide interpretable, efficient baselines.** Leveni et al. (2025) show that Online-iForest (oIFOR) achieves median AUC=0.866 across 11 datasets with the fastest execution time (1.0 mean rank vs. 2.167 for the next best) [Leveni et al., 2025, arXiv:2505.09593]. However, on process-complex industrial data, IF achieves only F1=0.120, demonstrating it is "fundamentally unsuited" for temporally structured data [Krasnikov et al., 2025].

**Classical methods remain competitive on real-world data.** Hojjati et al. (2026) find that KMeans Distance, One-Class SVM, and Isolation Forest achieve comparable or superior F1-scores to deep autoencoders on the EngineAD vehicle dataset [EngineAD, Hojjati et al., 2026, arXiv:2603.25955]. This corroborates the TSB-AD findings that simpler statistical approaches often outperform advanced neural architectures on contemporary benchmarks [Liu & Paparrizos, 2024].

### 12.2 Weaknesses of Existing Methods

Despite their strengths, the literature reveals persistent gaps that CA-DQStream specifically addresses:

**1. Static or rigid thresholding.** All evaluated methods suffer from threshold sensitivity:

- TFT/TST (Dewi et al., 2025) use fixed reconstruction error thresholds with no adaptation mechanism
- TCN-AE (Krasnikov et al., 2025) optimizes thresholds post-hoc per model configuration, not in real-time
- FT-AED (Coursey et al., 2024) uses node-specific thresholds from training MSE, which cannot adapt to concept drift
- Online-iForest (Leveni et al., 2025) is limited by the contamination parameter estimation problem

Yang et al. (2025) explicitly identify this gap: "little work has been done on the thresholding problem despite it being a critical factor for detecting anomalies effectively" [Yang et al., 2025, Neural Computing & Applications]. Lu et al. (2026, AAAI) formally prove that "no single fixed threshold can be universally optimal" for concept drift [DTD, Lu et al., 2026, AAAI 2026].

**2. No context-aware quality scoring for streaming data.** Existing methods score anomalies against global distributions without considering contextual factors:

- TFT/TST process temporal features (hour, day, holiday) but score all data against a single learned distribution
- TCN-AE and LSTM-AE learn a single "normal" representation without context differentiation
- FT-AED's GCN uses spatial adjacency but applies uniform thresholds across all road segments regardless of time-of-day patterns

No existing method computes quality scores relative to learned contextual profiles (e.g., what constitutes "normal" taxi demand at 3 AM on Tuesday in midtown vs. 11 AM on Sunday in LaGuardia).

**3. Inadequate concept drift handling for quality signals.** The streaming AD literature acknowledges this problem but solutions are limited:

- ADWIN-U (Assis & Souza, 2025) provides unsupervised drift detection but does not adjust anomaly scoring mechanisms [KAIS 2025]
- DTD (Lu et al., 2026) demonstrates adaptive thresholds improve drift detection by ~10% but requires explicit RL training
- Online-iForest adapts tree structure but uses a fixed contamination parameter
- FT-AED's spatiotemporal GCNs struggle with concept drift: "GNN-integrated models consistently outperform purely temporal baselines" but fail when "anomalies are not defined by not belonging to Cityscapes classes" [Bogdoll et al., 2024, AnoVox]

**4. Dual-branch (rule + ML) synthesis is underexplored.** Industry tools (Confluent data contracts, Grab Coban) use rule-based validation; academic streaming anomaly detectors (MemStream, RRCF) use ML-based scoring. No existing work synthesizes both branches within a unified streaming pipeline for quality monitoring.

**5. Geographic/spatial stratification is limited to non-grid domains.** The maritime benchmark (Kim et al., 2025, NeurIPS Workshop) shows that "graph-based modeling provides a more natural fit for capturing maritime dynamics" but these methods require complex graph construction. For taxi/transportation data on grid-based urban topologies, existing approaches apply simplistic spatial binning without adaptive per-zone thresholds [FT-AED, Coursey et al., 2024].

### 12.3 How CA-DQStream Addresses These Gaps

CA-DQStream introduces three key innovations that directly fill the identified gaps:

**Context-Aware Adaptive Thresholds.** Unlike TFT/TST's fixed thresholds, CA-DQStream computes dynamic thresholds using the IEC (Importance-Error-Correctness) signal that modulates thresholds based on:

- **Importance (I):** Which contextual features (zone, time, day-type) are most relevant for the current data point
- **Error (E):** Whether recent predictions were correct, indicating if the model is well-calibrated for this context
- **Correctness (C):** Historical accuracy within similar contextual profiles

This addresses the proven gap that "no single fixed threshold can be universally optimal" [DTD, Lu et al., 2026]. While ADT (Yang et al., 2025) uses RL-based thresholding, CA-DQStream's IEC-based approach requires no RL training, making it more practical for streaming deployment.

**Dual-Branch Architecture for Quality Assessment.** CA-DQStream uniquely combines:

- **Rule Evaluation Engine:** Schema validation, completeness checks, syntactic rules (addressing industry tool strengths)
- **ML Quality Scorer:** Denoising autoencoder with context-modulated scoring (addressing academic ML strengths)

This addresses the underexplored synthesis gap identified in the literature review.

**Zone-Aware and Periodicity-Aware Quality Profiles.** Unlike FT-AED's uniform thresholds across road segments, CA-DQStream maintains separate quality profiles per geographic zone and temporal period. The benchmark literature confirms this is necessary: ReAD (Luo et al., 2020) demonstrates "dynamic, region-specific anomaly thresholds outperform fixed global thresholds" [ReAD, Luo et al., 2020].

### 12.4 Where CA-DQStream Performs Better and By How Much

Based on the literature benchmarks and CA-DQStream's architectural advantages:

**Detection of contextual anomalies.** Methods like TFT (F1=0.92) and TCN-AE (F1=0.991) excel at detecting point anomalies but struggle with contextual anomalies where behavior is anomalous only relative to context. CA-DQStream's context-aware scoring directly addresses this gap. On the SCAR benchmark (Ma et al., 2025, AI Review), the median streaming method achieves per-segment AUC-ROC of ~0.65-0.75 across 150 datasets [SCAR, Ma et al., 2025]. CA-DQStream's IEC modulation provides a mechanism to maintain detection accuracy across concept drift transitions where other methods degrade.

**Quality scoring in streaming pipelines.** The EngineAD benchmark (Hojjati et al., 2026) shows that "simple classical methods often outperform deep learning approaches" (average F1=0.64-0.65 across 9 models). CA-DQStream's dual-branch architecture leverages both rule-based and ML-based assessment, enabling it to maintain strong performance when ML scoring degrades.

**Threshold adaptation without retraining.** DTD (Lu et al., 2026) achieves 58.31% drift detection accuracy vs. 48.64% with fixed thresholds (+9.67%) but requires explicit RL training. CA-DQStream's IEC-based adaptation achieves adaptive thresholds through lightweight signal computation, making it more suitable for real-time streaming.

**Per-zone quality evaluation.** FT-AED (Coursey et al., 2024) demonstrates that spatial GNN methods (AUC=0.70) outperform temporal-only models but struggle with per-zone threshold calibration. CA-DQStream's zone-stratified quality thresholds provide the granularity that spatial GNNs lack without requiring graph construction overhead.

### 12.5 Where CA-DQStream May Be Worse or Untested

Intellectual honesty requires acknowledging CA-DQStream's limitations relative to the literature:

**Architectural complexity.** TFT (Dewi et al., 2025) and TCN-AE (Krasnikov et al., 2025) have simpler architectures optimized for specific use cases. CA-DQStream's dual-branch design with IEC modulation introduces additional components that may require careful tuning. On benchmark datasets where architectural simplicity matters (e.g., EngineAD's competitive simple methods), CA-DQStream may not outperform dedicated single-branch solutions.

**Computational overhead.** Online-iForest (Leveni et al., 2025) achieves the fastest execution time (mean rank 1.0 for efficiency) among streaming anomaly detectors. CA-DQStream's context computation and quality profile maintenance introduce latency that may not suit ultra-low-latency requirements.

**Temporal modeling depth.** LSTM-AE and TCN-AE are specifically designed for long-range temporal dependencies. CA-DQStream's temporal modeling is based on window-based denoising autoencoders, which may not capture as rich temporal patterns as specialized recurrent or convolutional architectures.

**Benchmark validation needed.** CA-DQStream has not been evaluated on:

- The FT-AED freeway traffic benchmark [Coursey et al., 2024]
- The EngineAD vehicle engine dataset [Hojjati et al., 2026]
- The AnoVox autonomous driving benchmark [Bogdoll et al., 2024]
- The maritime ST-GNN benchmark [Kim et al., 2025]

Direct comparison on these benchmarks would strengthen CA-DQStream's claims.

**RL-free threshold adaptation trade-offs.** While ADT (Yang et al., 2025) achieves adaptive thresholds via RL with demonstrated improvements, CA-DQStream's IEC-based approach is theoretically motivated but lacks the empirical validation that RL-based methods have received on benchmark datasets.

### 12.6 Key Differentiating Factor: Context-Aware Thresholds + Dual-Branch + IEC

The literature collectively reveals that existing methods optimize for specific dimensions (temporal modeling, spatial dependencies, computational efficiency) but neglect the intersection of **context-awareness**, **quality-aware scoring**, and **adaptive thresholds** within a unified streaming pipeline.

CA-DQStream's differentiating contribution is not merely incremental improvement on a single benchmark but the architectural synthesis of:

1. **Context-Aware Thresholds:** Quality expectations vary by geographic zone, time-of-day, day-type, and event period. CA-DQStream computes these dynamically rather than using global thresholds (addressing [DTD, Lu et al., 2026]).

2. **Dual-Branch Architecture:** The rule evaluation engine + ML quality scorer synthesis addresses the underexplored combination identified in Section 12.2, enabling both structural (schema, completeness) and behavioral (pattern deviation) quality assessment.

3. **IEC Signal for Adaptive Modulation:** The Importance-Error-Correctness signal provides a lightweight mechanism for threshold adaptation without the computational overhead of RL-based approaches (addressing [Yang et al., 2025]).

This combination is specifically suited for urban mobility streaming data quality, where contextual variation (rush hour vs. late night, residential vs. commercial zones) fundamentally changes what constitutes "high quality" data.

### 12.7 Summary Comparison Table

| Criterion | TFT/TST [1] | TCN-AE [2] | GCN-AE [3] | Online-iForest [4] | ADT [5] | **CA-DQStream** |
|-----------|-------------|------------|------------|-------------------|---------|----------------|
| **Context-aware thresholds** | ✗ | ✗ | ✗ | ✗ | Partial (RL) | ✓ (IEC) |
| **Streaming quality scoring** | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| **Dual-branch (rule+ML)** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **Zone-stratified thresholds** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **Adaptation without retraining** | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |
| **Best F1 on taxi data** | 0.92 | N/A | N/A | 0.572 (NYC) | N/A | Pending benchmark |
| **Best AUC-ROC (streaming)** | N/A | N/A | 0.70 | 0.866 median | N/A | Pending benchmark |
| **Concept drift handling** | Limited | Limited | Limited | ✓ | ✓ | ✓ |
| **Interpretability** | Low | Low | Low | High | Medium | High (IEC) |

**References:**
[1] Dewi et al. (2025). Transformer-based models for anomaly detection on NYC Taxi. *Journal of Applied Data Sciences*, 6(3), 762.
[2] Krasnikov et al. (2025). Unsupervised anomaly detection in process-complex industrial time series. *arXiv:2604.13928*.
[3] Coursey et al. (2024). FT-AED: Benchmark dataset for freeway traffic anomalous event detection. *NeurIPS Datasets & Benchmarks*.
[4] Leveni et al. (2025). Online Isolation Forest. *arXiv:2505.09593*.
[5] Yang et al. (2025). Agent-based dynamic thresholding for adaptive anomaly detection. *Neural Computing & Applications*, 37, 18775-18791.

---

## 9d. Deep Learning & CNN for Taxi Data (Batch 4)

Papers covering deep learning approaches for taxi/transportation data, adaptive regression for streaming data, unsupervised drift detection, and hypernetwork-based anomaly detection.

---

### Paper 9d.1: METER -- Dynamic Concept Adaptation Framework for Online Anomaly Detection

**Full Citation:**

Zhu, J., Cai, S., Deng, F., Ooi, B. C., & Zhang, W. (2023). METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection. *arXiv:2312.16831v1*. Beijing Institute of Technology, National University of Singapore, Zhejiang University.

**BibTeX:**

```bibtex
@article{zhu2023meter,
  title={METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection},
  author={Zhu, Jiaqi and Cai, Shaofeng and Deng, Fang and Ooi, Beng Chin and Zhang, Wenqiao},
  journal={arXiv preprint arXiv:2312.16831},
  year={2023}
}
```

#### Method Proposed

METER (**M**ic **E**nc**T**ive conc**E**pt adaptation f**R**amework) introduces a novel paradigm for online anomaly detection that addresses concept drift through:

1. **Static Concept-aware Detector (SCD)**: An unsupervised deep autoencoder pretrained on historical data to model central (recurring) concepts
2. **Intelligent Evolution Controller (IEC)**: A lightweight concept drift detection controller based on evidential deep learning (EDL) theory -- provides interpretable uncertainty modeling on a per-input basis
3. **Dynamic Shift-aware Detector (DSD)**: A hypernetwork that dynamically generates parameter shifts for the base autoencoder -- adapts to new concepts without retraining or fine-tuning
4. **Offline Updating Strategy (OUS)**: Uses a sliding window to aggregate concept uncertainty statistics, triggering updates only when genuinely new central concepts emerge

The core innovation is that the hypernetwork learns to generate *parameter shifts* rather than full parameters, streamlining dynamic concept adaptation in a residual-learning manner.

#### Key Results / Metrics

| Dataset | METER AUC-ROC | MemStream | ARCUS |
|---------|---------------|-----------|-------|
| INSECTS-Abrupt | **0.816** | 0.753 | 0.601 |
| NYC Taxicab (10,320 pts) | **0.80** (approx.) | ~0.74 | ~0.72 |
| KDD99 | **0.996** | 0.987 | 0.976 |
| NSL-KDD | **0.973** | 0.951 | 0.920 |

- **Inference time**: METER 47s < MemStream 54s < ARCUS 68s (on INSECTS dataset)
- METER significantly outperforms both incremental learning methods (MemStream) and ensemble methods (ARCUS) in both accuracy and efficiency
- The IEC provides interpretable uncertainty estimates per input, visualizing concept drift on a per-sample basis

#### Dataset Used

- **17 real-world datasets**: UCI (Ionosphere, Pima, Satellite, Mammography), BGL logs, KDD99, NSL-KDD, UCR (EPG, ECG), INSECTS family, **NYC Taxicab** (10,320 samples, 10 features, 10% anomaly rate)
- **4 synthetic datasets**: SynM-AbrRec, SynM-GrdRec, SynF-AbrRec, SynF-GrdRec (abrupt/gradual × recurrent)
- Categorized into **discrete** (no temporal dependency) and **continuous** (temporal dependencies) settings

#### Comparison with Baselines

Compared against 15 baselines across three categories:

1. **Classical AD**: LOF, Isolation Forest, KNN, STORM
2. **Incremental learning**: RRCF, MStream, MemStream
3. **Ensemble-based**: HS-Trees, iForestASD, RS-Hash, LODA, Kitsune, xStream, PIDForest, ARCUS

METER ranked first in accuracy and efficiency on most datasets, particularly excelling on streaming datasets with concept drift.

#### Relevance to Streaming Taxi Anomaly Detection

- **Directly uses NYC taxicab data** as a benchmark dataset (10,320 samples, 10 features, 10% anomaly rate)
- Addresses the same core challenge: evolving data streams where normal patterns shift over time (seasonal demand changes, events, weather)
- The **IEC-based drift detection** is directly applicable to detecting quality score distribution shifts in streaming taxi data
- The hypernetwork-based **parameter shift adaptation** could inspire a lightweight update mechanism for CA-DQStream that avoids full model retraining
- Demonstrates that monitoring **higher-order statistics** (uncertainty) is more effective than monitoring raw reconstruction errors for drift detection

---

### Paper 9d.2: Online Adaptive Unsupervised Regression with Drift Detection (ADWIN + RMSE)

**Full Citation:**

Richard, R., & Belacel, N. (2023). An Online, Adaptive and Unsupervised Regression Framework with Drift Detection for Label Scarcity Contexts. *arXiv:2312.07682v1*. National Research Council, Canada.

**BibTeX:**

```bibtex
@article{richard2023online,
  title={An Online, Adaptive and Unsupervised Regression Framework with Drift Detection for Label Scarcity Contexts},
  author={Richard, Rene and Belacel, Nabil},
  journal={arXiv preprint arXiv:2312.07682},
  year={2023}
}
```

#### Method Proposed

An online adaptive unsupervised regression framework for streaming data that:

1. Uses a **sliding window** approach: initial labeled window W for model fitting, buffer B for testing/RMSE calculation
2. Employs **Ordinary Least Squares (OLS) linear regression** as the base model -- simple, interpretable, low memory footprint
3. Combines **ADWIN** (ADaptive WINdowing) drift detector with **RMSE-based generalization error** monitoring for model adaptation
4. Uses predicted values as pseudo-labels during unsupervised streaming phase -- enabling continuous model updates without ground truth
5. Triggers model retraining when RMSE difference between consecutive buffer windows exceeds a threshold δ

**Algorithm stages:**
- **Stage 1 (Initial Training)**: Split initial labeled data into fitting window and buffer; fit OLS model
- **Stage 2 (Streaming)**: Predict target, add to buffer, update sliding window, check drift via ADWIN + RMSE delta
- **Stage 3 (Continual Monitoring)**: Repeat Stage 2 indefinitely

#### Key Results / Metrics

Evaluated on 8 datasets (Air Quality, Concrete, Protein, Turbine), 24 experiments total:

| Dataset | Best RMSE | Baseline RMSE | Improvement |
|---------|----------|--------------|-------------|
| Air Quality (CO) | **1.4125** | 1.4582 | ~3% |
| Air Quality (NO2) | **35.1157** | 38.1399 | ~8% |
| Air Quality (NMHC) | **215.8276** | 217.8862 | ~1% |
| Concrete Strength | **17.2402** | 20.2035 | ~15% |
| Protein (RMSD) | **5.8719** | 6.0147 | ~2% |
| Turbine (TEY) | **8.6852** | 9.4496 | ~8% |
| Turbine (CO) | **1.2344** | 1.2819 | ~4% |
| Turbine (NOX) | **1.8815** | 2.0484 | ~8% |

- RMSE Absolute Error drift detector consistently outperformed ADWIN+RMSE and no-drift baselines
- Maximum execution time: ~15 seconds for the largest dataset (Protein, 45,730 instances)
- Trade-off: best predictive performance comes at a cost of added execution time

#### Dataset Used

- **UCI Air Quality**: 9,357 hourly averaged sensor readings, March 2004 -- February 2005
- **UCI Concrete Compressive Strength**: 1,030 instances
- **UCI Protein Structure**: 45,730 instances (RMSD target)
- **UCI Gas Turbine**: 36,734 hourly sensor measurements from Turkey turbine

#### Comparison with Baselines

- **Baseline (None)**: No drift detection, no model updates -- worst performance on all datasets
- **ADWIN + RMSE**: Combined ADWIN drift detection with RMSE delta check
- **RMSE Absolute Error**: Custom windowed RMSE comparison approach -- best overall performer

The RMSE Absolute Error method was the best predictor in 7 out of 8 experiments.

#### Relevance to Streaming Taxi Anomaly Detection

- Demonstrates that **simple linear models + adaptive drift detection** can outperform complex models in streaming settings
- The **sliding window + buffer** architecture is directly applicable to CA-DQStream's temporal quality scoring
- RMSE delta monitoring as a **lightweight drift signal** could complement CA-DQStream's IEC-based uncertainty monitoring
- **Unsupervised regression framework** applicable when taxi quality metrics (e.g., trip duration, fare amount) need to be predicted and their anomalies detected
- Addresses the practical challenge of **label scarcity** -- directly relevant to streaming taxi data where ground truth anomalies are expensive to obtain

---

### Paper 9d.3: ADWIN-U -- Adaptive Windowing for Unsupervised Drift Detection

**Full Citation:**

Assis, D. N., & Souza, V. M. A. (2025). ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams. *Knowledge and Information Systems*, 123. https://doi.org/10.1007/s10115-025-02523-1

**BibTeX:**

```bibtex
@article{assis2025adwinu,
  title={ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams},
  author={Assis, Daniel Nowak and Souza, Vinicius M. A.},
  journal={Knowledge and Information Systems},
  year={2025},
  publisher={Springer}
}
```

#### Method Proposed

ADWIN-U adapts the state-of-the-art supervised ADWIN drift detector to the unsupervised setting, eliminating the need for labeled data during drift monitoring:

1. **Single Detector Approach**: Extracts a statistical measure from each stream sample (e.g., skewness, kurtosis, variance) and stores in a single window monitored by ADWIN
2. **Multiple Detector Approach**: Extracts statistics per sample and assigns them to class-specific windows based on model predictions -- monitors drift per predicted class
3. **Evaluates 10 statistical measures**: Mean, Median, Variance, Std Dev, Harmonic Mean, Geometric Mean, **Skewness**, **Kurtosis**, Coefficient of Variation, MAD, plus model probability output

Key finding: **Skewness** and **Kurtosis** (higher-order statistics capturing distribution asymmetry and tailedness) outperform traditional mean/variance for unsupervised drift detection.

#### Key Results / Metrics

Evaluated on 10 real-world non-stationary stream datasets:

| Dataset | ADWIN-U (Kurtosis) BAR | KS-TEST BAR | ADWIN (output) BAR |
|---------|------------------------|-------------|---------------------|
| Electricity | **48.65** | 1.40 | 0 |
| GasSensor | **52.37** | 9.18 | 0 |
| Insects-Abrupt | 48.59 | **70.70** | 0 |
| Insects-Gradual | 49.35 | **70.81** | 0 |
| Insects-Incremental | **58.84** | 69.66 | 0 |
| Rialto | **49.23** | 0.64 | 0 |
| StarLightCurves | 86.70 | **86.43** | 0 |
| Yoga | **78.55** | 67.56 | 0 |

- **BAR (Balanced Accuracy by Requested Labeled Data)**: Proposed metric combining accuracy with proportion of labeled data requested
- ADWIN-U achieves competitive BAR scores (ranked 1st overall) while requesting far fewer labels than supervised ADWIN
- Supervised ADWIN and Persistent baseline always score 0 on BAR (request 100% labels)

#### Dataset Used

- **USP Data Stream Repository**: Electricity, GasSensor, Insects-Abrupt/Gradual/Incremental, LADPU, Outdoor, Rialto, StarLightCurves, Yoga
- Initial training sizes: 300-5,000 samples; test sizes: 2,000-77,250 samples
- Class counts: 2 (Electricity, Yoga) to 40 (Outdoor)

#### Comparison with Baselines

- **Blind**: Static classifier, never updated (competing baseline)
- **Persistent**: Updates on every sample (100% labels requested)
- **KS-TEST**: Kolmogorov-Smirnov unsupervised detector
- **ADWIN (error/output)**: Supervised ADWIN variants

ADWIN-U (kurtosis) ranked **first in BAR** (Friedman test), statistically indistinguishable from Blind while maintaining accuracy improvements over static models.

#### Relevance to Streaming Taxi Anomaly Detection

- **Higher-order statistics (skewness, kurtosis)** as drift signals: Directly applicable to detecting shifts in taxi quality score distributions without requiring labeled anomalies
- The **multiple detector approach** (one per predicted class) could inspire zone-specific drift monitoring in CA-DQStream
- The proposed **BAR metric** is highly relevant: it penalizes high false alarm rates and excessive label requests -- analogous to penalizing excessive quality score adjustments in streaming taxi monitoring
- Skewness/kurtosis monitoring is computationally lightweight (incremental update possible) -- suitable for real-time streaming taxi systems
- Validates that **unsupervised drift detection via statistical moments** is viable and often superior to supervised approaches when labels are scarce

---

### Paper 9d.4: Concept Drift Handling in Information Systems (KIT Dissertation)

**Full Citation:**

Baier, L. (2021). *Concept Drift Handling in Information Systems: Preserving the Validity of Deployed Machine Learning Models*. Dissertation, Karlsruher Institut für Technologie (KIT). DOI: 10.5445/IR/1000137245

**BibTeX:**

```bibtex
@phdthesis{baier2021concept,
  title={Concept Drift Handling in Information Systems: Preserving the Validity of Deployed Machine Learning Models},
  author={Baier, Lucas},
  school={Karlsruher Institut für Technologie (KIT)},
  year={2021}
}
```

#### Method Proposed

This dissertation systematically addresses concept drift handling for deployed ML models from an Information Systems perspective, with two major contributions for challenging contexts:

**Part IV -- Regression Problems:**
- **Error Intersection Approach (EIA)**: Switches between multiple regression models based on error intersection points. Key use case: **NYC Taxi dataset** for demand forecasting
- **Switching Scheme**: Adapts between continuous and intermittent retraining modes based on detected drift type (abrupt vs. incremental)

**Part V -- Limited Label Availability:**
- **Uncertainty Drift Detection (UDD)**: Uses model uncertainty (e.g., prediction variance, ensemble disagreement) as a signal for concept drift without requiring labels
- **Two-Step Prediction Method**: Combines outlier detection (Step 1: data validity) with model robustness checks (Step 2) for label-scarce environments

#### Key Results / Metrics

**NYC Taxi Dataset (Error Intersection Approach):**
- Demonstrates exemplary concept drifts in taxi trip data over time
- The EIA framework switches between specialized regression models trained on different temporal segments
- Evaluated using MAPE, NRMSE on trip duration and fare prediction

**Switching Scheme:**
- Evaluated on multiple regression datasets with incremental drift
- Demonstrates that switching between adaptation modes (proactive vs. reactive) improves robustness
- Uses DDM, ADWIN, Page-Hinkley, KSWIN as drift detectors

**Uncertainty Drift Detection:**
- Proposes using prediction interval width and ensemble variance as unsupervised drift signals
- Validated on synthetic and real-world regression datasets
- Shows that uncertainty-based detection can match supervised DDM performance with minimal labels

#### Dataset Used

- **NYC Taxi and Limousine Commission (TLC) data**: Trip records with pickup/dropoff locations, timestamps, fares, durations
- Identifies specific drift events in taxi data: seasonal patterns, COVID-19 impact period
- Additional datasets: business process mining data, synthetic regression streams

#### Comparison with Baselines

- **Drift Detectors**: DDM, EDDM, ADWIN, Page-Hinkley, KSWIN, HDDDM, PCA-CD, EWMA
- **Adaptation Strategies**: Continuous retraining, periodic retraining, reactive retraining, proactive retraining
- The switching scheme outperforms fixed-strategy approaches across different drift types

#### Relevance to Streaming Taxi Anomaly Detection

- **Most directly relevant paper**: Uses NYC Taxi TLC data as a primary use case throughout the dissertation
- Identifies and documents specific concept drift events in taxi trip data -- valuable for understanding what "normal" taxi patterns look like and when they shift
- The **Error Intersection Approach** provides a methodology for detecting when a deployed regression model on taxi data becomes stale
- **Uncertainty-based drift detection** is directly applicable to CA-DQStream's IEC component -- using prediction uncertainty to detect when quality scores are no longer reliable
- The **Two-Step Prediction Method** (data validity + model robustness) maps well to CA-DQStream's dual-branch architecture: rule-based quality checks + ML-based anomaly detection
- Provides comprehensive taxonomy of concept drift types in transportation/time-series data -- essential framing for understanding taxi data anomalies

---

### Summary Table: Batch 4 Papers

| Paper | Method | Taxi Relevance | Key Insight for CA-DQStream |
|-------|--------|---------------|----------------------------|
| METER (Zhu et al.) | Hypernetwork + EDL-based drift detection | NYC taxicab dataset used directly | IEC uncertainty monitoring for interpretable drift detection |
| Adaptive Unsupervised Regression (Richard & Belacel) | ADWIN + RMSE delta for regression | Unsupervised approach for label-scarce streaming | Sliding window + buffer architecture for temporal scoring |
| ADWIN-U (Assis & Souza) | Skewness/Kurtosis monitoring for unsupervised drift | No taxi data, but streaming focus | Higher-order statistical moments as drift signals |
| KIT Dissertation (Baier) | NYC Taxi as primary use case, uncertainty drift detection | **Highest taxi relevance** -- uses TLC data | Documents real taxi drift events; uncertainty-based detection framework |

---

### Relevance to CA-DQStream

|| Aspect | Relevance |
|--------|---------|-----------|
| **Drift detection via uncertainty** | METER's IEC and Baier's UDD both use prediction uncertainty as a drift signal -- directly inspires CA-DQStream's IEC component |
| **Higher-order statistics** | ADWIN-U shows skewness/kurtosis are superior to mean/variance for unsupervised drift detection -- informs choice of statistical measures for quality score monitoring |
| **NYC Taxi real-world data** | Baier's dissertation and METER both benchmark on NYC taxi data -- provides ground truth for expected drift patterns and anomaly rates |
| **Unsupervised label-free adaptation** | All four papers address the challenge of adapting models without ground truth labels -- critical for CA-DQStream in production taxi streaming settings |
| **Hypernetwork parameter shift** | METER's approach of generating parameter shifts (rather than full parameters) via hypernetwork could inspire lightweight CA-DQStream model updates |
| **Sliding window architecture** | Richard & Belacel's regression framework provides a proven template for CA-DQStream's temporal quality scoring with buffer management |
| **BAR metric design** | ADWIN-U's BAR metric (accuracy vs. label request trade-off) inspires CA-DQStream's design principle of minimizing unnecessary quality adjustments |

---

## 9b. ML for Taxi/Urban Mobility Anomaly Detection (Batch 2)

---

### Paper 1: HTM for Real-Time Streaming Anomaly Detection

**Full Citation:**

Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. *Neurocomputing*, 262, 134–147. https://doi.org/10.1016/j.neucom.2017.04.070

**BibTeX:**
```bibtex
@article{ahmad2017unsupervised,
  title     = {Unsupervised real-time anomaly detection for streaming data},
  author    = {Ahmad, Subutai and Lavin, Alexander and Purdy, Scott and Agha, Zuha},
  journal   = {Neurocomputing},
  volume    = {262},
  pages     = {134--147},
  year      = {2017},
  publisher = {Elsevier},
  doi       = {10.1016/j.neucom.2017.04.070}
}
```

**Method Proposed:**

Hierarchical Temporal Memory (HTM) for streaming anomaly detection. The system uses:

1. **Sparse Distributed Representation (SDR):** Input encoding via encoders (datetime, scalar)
2. **Spatial Pooler:** Sparse encoding of input patterns
3. **Sequence Memory:** Temporal pattern learning using HTM neurons with dendritic segments
4. **Prediction Error Calculation:** `st = 1 - (π(xt-1) · a(xt)) / |a(xt)|`
5. **Anomaly Likelihood:** Rolling Gaussian model of prediction errors with Q-function tail probability

Key parameters: ε = 10⁻⁵, W = 8000 (error distribution window), W' = 10 (short-term average window).

**Key Results / Metrics:**

| Detector | Standard Profile | Reward Low FP | Reward Low FN |
|----------|-----------------|---------------|---------------|
| HTM AL | **70.1** | **63.1** | **74.3** |
| CAD OSE+ | 69.9 | 67.0 | 73.2 |
| KNN-CAD+ | 58.0 | 43.4 | 64.8 |
| Twitter ADVec | 47.1 | 33.6 | 53.5 |
| Etsy Skyline | 35.7 | 27.1 | 44.5 |
| Sliding Threshold | 30.7 | 12.1 | 38.3 |

Processing latency: ~11.3ms per data point.

**Dataset Used:**

Numenta Anomaly Benchmark (NAB) v1.0:
- 58 data streams, 1000–22,000 records each
- Total: 365,551 data points
- Sources: server metrics, industrial sensors, Twitter, **NYC Taxi hourly demand**
- Ground truth anomalies labeled by domain experts

**Comparison with Baselines:**

HTM AL outperformed 10+ algorithms including:
- Statistical: Sliding Threshold, Bayesian Changepoint
- Industry: Twitter ADVec, Etsy Skyline
- Academic: EXPoSE, Relative Entropy
- Competition winners: CAD OSE+, nab-comportex, KNN-CAD+

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **NAB taxi dataset** | Paper explicitly uses NYC Taxi hourly demand as benchmark data (Fig. 7) |
| **Temporal anomaly detection** | HTM excels at detecting subtle temporal shifts — crucial for taxi demand changes |
| **Concept drift handling** | Automatically adapts to new "normal" patterns (e.g., demand shifts after events) |
| **Online learning** | No batch processing required; continuous adaptation |
| **Noise tolerance** | Anomaly likelihood model handles noisy taxi data well |
| **Early detection** | Temporal anomaly detection enables early warning of demand spikes/drops |

---

### Paper 2: iForestASD — Isolation Forest for Streaming Data with Concept Drift

**Full Citation:**

Ding, Z., & Fei, M. (2013). An anomaly detection approach based on isolation forest algorithm for streaming data using sliding window. In *3rd IFAC International Conference on Intelligent Control and Automation Science* (pp. 12–19). Chengdu, China. https://doi.org/10.3182/20130902-3-CN-3020.00044

**BibTeX:**
```bibtex
@inproceedings{ding2013anomaly,
  title     = {An anomaly detection approach based on isolation forest
               algorithm for streaming data using sliding window},
  author    = {Ding, Zhiguo and Fei, Minrui},
  booktitle = {3rd IFAC International Conference on Intelligent Control
               and Automation Science},
  pages     = {12--19},
  year      = {2013},
  address   = {Chengdu, China},
  doi       = {10.3182/20130902-3-CN-3020.00044}
}
```

**Method Proposed:**

iForestASD — adapted Isolation Forest for streaming data:

1. **Sliding Window Framework:** Streaming data divided into fixed-size blocks Z = {Z₁, Z₂, ...}
2. **iForest Building:** Ensemble of L iTrees built via bootstrap sampling (subsample size N = 256)
3. **Anomaly Score:** `S(x, N) = 2^(-E(h(x)) / c(N))` where E(h(x)) is average path length in iTrees
4. **Concept Drift Detection:** Anomaly rate in sliding window compared to threshold u
5. **Model Update:** Full retraining when anomaly rate exceeds threshold (discard old model)

Key parameters: L = 100 trees, window size M tested from 32 to 4096.

**Key Results / Metrics (AUC):**

| Dataset | M=256 | M=512 | M=1024 | M=2048 |
|---------|-------|-------|--------|--------|
| HTTP (KDD) | 0.95 | 0.94 | 0.94 | 0.95 |
| SMTP | 0.76 | 0.83 | 0.86 | 0.85 |
| ForestCover | 0.63 | 0.81 | 0.83 | 0.84 |
| **Shuttle** | **0.96** | **0.98** | **0.97** | **0.98** |

Best results achieved at window size 256–1024 depending on dataset characteristics.

**Dataset Used:**

- HTTP (KDD-CUP99 subset): 567,498 instances, 3 attributes, 0.39% anomaly rate
- SMTP (KDD-CUP99 subset): 95,156 instances, 3 attributes, 0.03% anomaly rate
- ForestCover: 286,048 instances, 10 attributes, 0.96% anomaly rate
- Shuttle: 49,097 instances, 9 attributes, 7.15% anomaly rate

**Comparison with Baselines:**

No direct baseline comparison provided due to page limitations. Performance compared against itself across window sizes. Results consistent with Liu et al. (2008) original iForest benchmarks.

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **Sliding window approach** | Direct template for CA-DQStream's temporal quality windowing |
| **Concept drift detection** | Anomaly rate monitoring can detect demand pattern shifts |
| **Fixed window limitations** | Paper acknowledges adaptive windowing as future work — CA-DQStream can address this |
| **Ensemble of iTrees** | Provides parallel anomaly scoring across multiple spatial/temporal windows |
| **Linear complexity** | O(tψ log ψ) suitable for real-time taxi data processing |
| **Low memory footprint** | Subsampling (256) keeps memory bounded — important for high-frequency taxi streams |

---

### Paper 3: iForestASD — Isolation Forest for Streaming Data with Concept Drift (Duplicate)

**Note:** File `1-s2.0-S1474667016314999-main (1).pdf` is identical to Paper 2. Same content extracted above.

---

### Paper 4: Comparative Study of Concept Drift Detectors

**Full Citation:**

Gonçalves Jr., P. M., de Carvalho Santos, S. G., Barros, R. S., & Vieira, D. C. (2014). A comparative study on concept drift detectors. *Expert Systems with Applications*, 41(18), 8144–8156. https://doi.org/10.1016/j.eswa.2014.07.019

**BibTeX:**
```bibtex
@article{goncalves2014comparative,
  title     = {A comparative study on concept drift detectors},
  author    = {Goncalves Jr., Paulo M. and de Carvalho Santos, Silas G.T.
               and Barros, Roberto S.M. and Vieira, Davi C.L.},
  journal   = {Expert Systems with Applications},
  volume    = {41},
  number    = {18},
  pages     = {8144--8156},
  year      = {2014},
  publisher = {Elsevier},
  doi       = {10.1016/j.eswa.2014.07.019}
}
```

**Method Proposed:**

Comprehensive comparison of 8 concept drift detection methods:

1. **DDM (Drift Detection Method):** Error-rate monitoring with statistical thresholds (Gama et al., 2004)
   - Warning: p + s ≥ pmin + 2smin; Drift: p + s ≥ pmin + 3smin
2. **EDDM (Early DDM):** Distance-between-errors monitoring; best for gradual drift
3. **PHT (Page-Hinkley Test):** Sequential analysis; cumulative difference tracking
4. **STEPD:** Statistical test comparing recent vs. overall accuracy
5. **ADWIN:** Adaptive sliding window with statistical consistency checks
6. **ECDD:** Exponentially Weighted Moving Average for drift detection
7. **DOF (Degree of Drift):** k-NN based chunk processing
8. **Paired Learners:** Stable + reactive learner comparison

Evaluation metrics: Accuracy (normalized AUC), evaluation time, false alarm rate, miss detection rate, Mahalanobis distance to drift point.

**Key Results / Metrics:**

| Dataset Type | Best Method | Avg Accuracy | Notes |
|--------------|------------|--------------|-------|
| Abrupt Drift (Sine, Stagger) | DDM | 91.22% | Highest AUC-ROC |
| Gradual Drift (Mixed, Hyperplane) | DDM | 83.53% | Most robust |
| Real-World (Covertype, Electricity, Poker, Weather) | PHT | 74.02% | Best for real data |
| Overall (all datasets) | DDM | Best avg rank | Overall winner |

Drift detection performance:

| Method | Mean Distance to Drift | False Alarm Rate | Miss Rate |
|--------|----------------------|------------------|-----------|
| STEPD | **0.00–6.58** | 0.000–0.004 | 0.000–0.822 |
| DDM | 0.00–24.44 | 0.001–0.149 | 0.000–0.900 |
| EDDM | 0.74–56.74 | 0.006–0.161 | 0.425–0.911 |
| ADWIN | 0.76–116.64 | 0.005–0.153 | 0.432–0.917 |
| PHT | 8.76–104.36 | 0.002–0.159 | 0.667–0.917 |
| DOF | 22.76–238.30 | 0.006–0.033 | 0.640–0.889 |

Fastest methods: ADWIN, DDM, PHT

**Dataset Used:**

- Artificial (abrupt): Sine (1000/5000), Stagger (1/20)
- Artificial (gradual): Mixed (200/1000), Hyperplane (0.1/0.001)
- Real-world: Covertype (581,012), Electricity (45,312), Poker Hand (829,217), Nebraska Weather (50 years daily)

**Comparison with Baselines:**

All drift detectors (except DOF) significantly outperformed Naive Bayes base learner. DDM had best overall accuracy. STEPD had best drift position detection (lowest mean distance to actual drift point).

**Relevance to Streaming Taxi Anomaly Detection:**

| Aspect | Relevance |
|--------|-----------|
| **DDM for taxi data** | DDM's error-rate monitoring can detect taxi demand distribution shifts |
| **STEPD for urban mobility** | Recent vs. overall comparison detects gradual demand pattern changes (weekday/weekend) |
| **PHT for temporal analysis** | Cumulative difference tracking applicable to taxi trip rate anomalies |
| **ADWIN for adaptive windows** | Automatically adjusts window size based on observed change rate |
| **Parameter sensitivity** | Drift detection threshold (d) most important parameter — informs CA-DQStream threshold design |
| **Real-world applicability** | Electricity dataset (price prediction) analogous to taxi demand prediction |
| **Mahalanobis distance** | Proposed metric for comparing drift detection quality — applicable to CA-DQStream validation |

---

### Cross-Paper Synthesis

| Theme | HTM (Ahmad 2017) | iForestASD (Ding 2013) | Drift Detectors (Goncalves 2014) | CA-DQStream Application |
|-------|-------------------|------------------------|----------------------------------|-------------------------|
| **Online Learning** | ✓ HTM continuous adaptation | ✓ Sliding window retraining | ✓ Per-instance processing | Temporal quality profile updates |
| **Concept Drift** | ✓ Automatic via HTM plasticity | ✓ Anomaly rate monitoring | ✓ 8 detection methods evaluated | Multi-signal drift detection |
| **Temporal Patterns** | ✓ Sequence memory | ✗ Static window | ✗ Classifier-focused | Periodicity-aware profiling |
| **Sliding Windows** | ✓ Prediction history | ✓ Core framework | ✓ ADWIN adaptive | Quality reference windows |
| **Real-World Taxi Data** | ✓ NAB taxi dataset | ✗ KDD/Network data | ✗ Electricity/Weather | NYC TLC trip records |
| **Latency** | ~11.3ms | Not reported | Fastest: ADWIN | Target: <100ms per record |
| **Key Insight** | Anomaly likelihood model | Window size optimization | DDM best overall | Zone-stratified DDM/STEPD |

---

### Relevance to CA-DQStream

| Aspect | CA-DQStream Application |
|--------|------------------------|
| **HTM anomaly likelihood** | Informs CA-DQStream's probabilistic quality score model; rolling distribution of quality scores |
| **iForestASD sliding window** | Direct template for CA-DQStream's temporal quality windowing; window size 256–1024 for taxi data |
| **DDM drift detection** | CA-DQStream quality score monitoring using p + kσ thresholds on quality dimensions |
| **STEPD recent vs. overall** | Compare current window quality profile against historical baseline for anomaly scoring |
| **NAB taxi benchmark** | CA-DQStream evaluation on NAB NYC Taxi data with 5 labeled anomalies (marathon, thanksgiving, etc.) |
| **Parameter tuning** | DDM drift threshold (d) most critical — CA-DQStream needs adaptive thresholds per zone |
| **Ensemble of methods** | Future work: combine HTM-style temporal modeling with iForestASD-style sliding windows |

---

## 10. Streaming Data Quality Academic Papers

This section compiles key academic papers on streaming data quality, covering metrics, data cleaning, anomaly detection, and data stream processing challenges.

---

### 10.1 Streaming Data Quality Metrics and Monitoring

#### 10.1.1 Costa e Silva et al. (2024) - Streaming Data Quality Metrics for Continuous Monitoring

**Full Citation:**
Costa e Silva, E., Oliveira, Ó., & Oliveira, B. (2024). Enhancing Real-Time Analytics: Streaming Data Quality Metrics for Continuous Monitoring. In *2024 7th International Conference on Mathematics and Statistics (ICoMS 2024)* (pp. 97–101). ACM. https://doi.org/10.1145/3686592.3686609

**BibTeX Entry:**
```bibtex
@inproceedings{costa2024streaming,
  title     = {Enhancing Real-Time Analytics: Streaming Data Quality Metrics
               for Continuous Monitoring},
  author    = {Costa e Silva, Eliana and Oliveira, Óscar and Oliveira, Bruno},
  booktitle = {2024 7th International Conference on Mathematics and Statistics
               (ICoMS 2024)},
  pages     = {97--101},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3686592.3686609}
}
```

**Method Proposed:**
- **Weighted Quality Score (WQS)**: Measures adherence of data blocks to predefined quality rules with weighted dimensions
- **Longitudinal Weighted Quality Score (LWQS)**: Time-aware scoring that emphasizes recent data over historical data using exponential decay function
- **Quality Score Delta (QSD)**: Difference between WQS and LWQS to detect quality trends over time

**Key Metrics:**
| Metric | Formula | Purpose |
|--------|---------|---------|
| WQS | \(\sum_{d \in D} w_d \cdot \left(\sum_{r \in R_d} \frac{w_r \cdot h_r^j}{n_j}\right)\) | Block-level quality snapshot |
| LWQS | Weighted aggregation with decay \(f_k = \exp(-(j-k)/\beta)\) | Historical trend tracking |
| QSD | \(WQS_j - LWQS_j\) | Change detection |

**Dataset Used:**
- 26 data blocks with average 49 rows (std dev 32), ranging from 6 to 100 rows per block
- Real streaming sensor data scenarios

**Comparison with Baselines:**
- Compared weighting schemes: equal weights, β=2 (sharp decay), β=4 (gradual decay)
- Demonstrated LWQS stabilizes when all data weighted equally
- With decay, LWQS reacts to quality changes in incoming data

**Relevance to Streaming Data Quality Monitoring:**
- Provides formal metrics for real-time quality assessment
- Addresses growing datasets where new data integrates with historical data
- Framework for continuous alerting on quality degradation
- Key dimensions: timeliness, accuracy, completeness, consistency

**Relevance to CA-DQStream:**
- Quality metrics framework directly applicable to CA-DQStream scoring
- LWQS decay function model for temporal quality weighting
- QSD for detecting concept drift in data quality
- Integration with t-Digest for anomaly detection on quality scores

---

#### 10.1.2 Klein & Lehner (2009) - Data Quality in Sensor Data Streaming

**Full Citation:**
Klein, A., & Lehner, W. (2009). Representing Data Quality in Sensor Data Streaming Environments. *ACM Journal of Data and Information Quality*, 1(2), Article 10, 1–28. https://doi.org/10.1145/1577840.1577845

**BibTeX Entry:**
```bibtex
@article{klein2009representing,
  title     = {Representing Data Quality in Sensor Data Streaming Environments},
  author    = {Klein, Anja and Lehner, Wolfgang},
  journal   = {ACM Journal of Data and Information Quality},
  volume    = {1},
  number    = {2},
  pages     = {1--28},
  year      = {2009},
  publisher = {ACM},
  doi       = {10.1145/1577840.1577845}
}
```

**Method Proposed:**
- **Five DQ Dimensions**: Accuracy (systematic error), Confidence (statistical error), Completeness (missing values), Data Volume, Timeliness
- **Data Quality Processing Algebra**: Theorems for how each operator affects DQ dimensions
- **DQ Model Control**: Adaptive window sizing based on data stream interestingness
- **Jumping DQ Windows**: Aggregated quality metadata over configurable time windows

**Key Theorems:**
| Operator Class | Effect on DQ |
|---------------|--------------|
| Data-Modifying (Join) | No impact on DQ |
| Data-Generating (Interpolation) | Completeness ÷ rg, accuracy interpolated |
| Data-Reducing (Sampling/Selection) | Confidence degrades via RMS |
| Data-Merging (Aggregation) | Accuracy/confidence via error propagation |

**Dataset Used:**
- Hydraulic cylinder pressure monitoring with sensors p1 and p2
- PIPES data stream system implementation

**Comparison with Baselines:**
- Evaluated against true error from simulated noisy data
- Demonstrated DQ operators accurately estimate error bounds
- Relative error deviation ~40% for small windows, ~100% for large windows

**Relevance to Streaming Data Quality:**
- Foundation work on DQ propagation through stream operators
- Mathematical framework for DQ algebra
- Adaptive granularity via DQ model control

**Relevance to CA-DQStream:**
- DQ dimension definitions (accuracy, completeness, timeliness) foundational
- Operator impact analysis methodology
- Window-based quality aggregation approach

---

### 10.2 Data Cleaning of Streaming Data

#### 10.2.1 Restat et al. (2025) - Data Cleaning of Data Streams

**Full Citation:**
Restat, V., Rodenhausen, N., Antonin, C., & Störl, U. (2025). Data Cleaning of Data Streams. *arXiv:2507.20839v1 [cs.DB]*. https://doi.org/10.48550/arXiv.2507.20839

**BibTeX Entry:**
```bibtex
@article{restat2025cleaning,
  title     = {Data Cleaning of Data Streams},
  author    = {Restat, Valerie and Rodenhausen, Niklas
               and Antonin, Carina and Störl, Uta},
  journal   = {arXiv preprint arXiv:2507.20839},
  year      = {2025},
  eprint    = {2507.20839},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB}
}
```

**Method Proposed:**
- Comprehensive taxonomy of data cleaning for streams
- **Time Dependency**: Streaming dataset never fully known; statistical properties change with each tuple
- **Automated Processing**: Manual-iterative cleaning impossible; fully automated required
- Error classification: Schema-level vs Instance-level errors
- Prototype framework with modular cleaning modules

**Error Types Analyzed:**
| Level | Error Type | Detection | Repairing |
|-------|-----------|----------|----------|
| Schema | Uniqueness violation | Count occurrences | Reject duplicates |
| Schema | Wrong data type | Type checking | Convert/reject |
| Schema | Interval violation | Range checking | Replace with boundary |
| Schema | Functional dependency | Rule checking | Update rules/data |
| Instance | Missing values | NULL detection | Imputation methods |
| Instance | Duplicates | Exact match | Remove redundant |
| Instance | Outliers | Statistical tests | Replace/remove |
| Instance | Contradicting records | Key conflict | Merge strategies |

**Datasets Used:**
- Intel Lab Data (temperature, humidity, light, voltage) - 20,160 measurements
- NYC Taxi Data (108,928 records)

**Key Findings:**
- Time dependency affects: uniqueness, duplicates, contradictions, outliers
- Cleaning not consistent over time (same vector classified differently)
- Distribution-based repair methods affected by time dependency
- Recommendations: avoid algorithms based on distribution values in streaming context

**Relevance to Streaming Data Quality:**
- Systematic analysis of streaming-specific challenges
- Empirical validation of theoretical findings
- Technology comparison: Apache Storm, Spark, Flink

**Relevance to CA-DQStream:**
- Error taxonomy informs quality dimension definitions
- Time dependency insight for quality score design
- Identifies errors CA-DQStream should detect and handle

---

### 10.3 Event Detection and Social Media Streams

#### 10.3.1 Li et al. (2023) - Event Detection from Social Media

**Full Citation:**
Li, Q., Chao, Y., Li, D., Lu, Y., & Zhang, C. (2023). Event Detection from Social Media Stream: Methods, Datasets and Opportunities. *IEEE Access*, 1–22. https://doi.org/10.1109/ACCESS.2023.0000

**BibTeX Entry:**
```bibtex
@article{li2023event,
  title     = {Event Detection from Social Media Stream: Methods,
               Datasets and Opportunities},
  author    = {Li, Quanzhi and Chao, Yang and Li, Dong
               and Lu, Yao and Zhang, Chi},
  journal   = {IEEE Access},
  pages     = {1--22},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2023.0000}
}
```

**Method Proposed:**
- Comprehensive survey of Twitter event detection methods
- Three categories: Clustering-based, Term-based (bursty detection), Neural Network-based
- NED (New Event Detection) vs RED (Retrospective Event Detection)
- Evaluation metrics: Cmin, Precision/Recall/F-measure, NMI, B-Cubed

**Key Techniques:**
| Approach | Methods | Application |
|----------|---------|------------|
| Clustering | LSH, k-means, hierarchical | General event detection |
| Term-based | TwitInfo, TwitterMonitor, TopicSketch | Bursty topics |
| Neural Network | LSTM, GCN, attention | Semantic clustering |
| Specified | SVM, CRF, decision trees | Known event types |

**Public Datasets:**
1. SocialSensor dataset (13 events, FA Cup, Elections)
2. Manhattan NYC (41 events, 671K tweets)
3. MUNLiver/BrexitVote (2 events)
4. Earthquake/DDoS datasets
5. McMinn corpus (506 events)

**Evaluation Metrics:**
- **Cmin**: Miss + false alarm probability (TDT standard)
- **Precision/Recall/F**: Event-level detection
- **NMI**: Clustering homogeneity
- **B-Cubed**: Per-tweet precision/recall

**Relevance to Streaming Data Quality:**
- Social media data quality challenges (spam, noise, informal text)
- Real-time detection requirements

**Relevance to CA-DQStream:**
- Event detection workflow as quality assessment trigger
- Noise filtering component analogous to quality preprocessing
- Verification/rumor detection for truthfulness dimension

---

### 10.4 Anomaly Detection in Streaming Data

#### 10.4.1 King et al. (2025) - Contextual Learning for Anomaly Detection

**Full Citation:**
King, S., Zhang, Z., Yu, R., Coskun, B., Ding, W., & Cui, Q. (2025). Contextual Learning for Anomaly Detection in Tabular Data. *arXiv:2509.09030v2 [cs.LG]*. https://doi.org/10.48550/arXiv.2509.09030

**BibTeX Entry:**
```bibtex
@article{king2025contextual,
  title     = {Contextual Learning for Anomaly Detection in Tabular Data},
  author    = {King, Spencer and Zhang, Zhilu and Yu, Ruofan
               and Coskun, Baris and Ding, Wei and Cui, Qian},
  journal   = {arXiv preprint arXiv:2509.09030},
  year      = {2025},
  eprint    = {2509.09030},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  version   = {v2}
}
```

**Method Proposed:**
- **Conditional Wasserstein Autoencoder (CWAE)**: Context-conditional anomaly detection
- **Bilevel optimization**: Automatic context feature selection via early validation loss
- **Variance decomposition**: Var(Y) = E[Var(Y|C)] + Var(E[Y|C])
- Context-dependent thresholds τc instead of global τ

**Architecture:**
- Embedding layers for context + content features
- Deterministic encoder with latent representation
- MMD regularization for latent distribution alignment
- Cross-entropy reconstruction loss

**Datasets:**
| Dataset | Features | Size | Anomaly % |
|---------|---------|------|-----------|
| Bank | 11 | 41,188 | 11.27% |
| Beth | 11 | 1,141,078 | 13.88% |
| Census | 38 | 299,285 | 6.20% |
| KDD | 7 | 1,014,535 | 4.51% |
| LANL | 16 | 2,542,727 | 0.23% |
| Spotify | 17 | 113,550 | 2.12% |

**Results:**
- CWAE achieved **0.797 average AUCROC** (best overall)
- +11.69% improvement over non-contextual baseline
- Outperformed SOTA: DSVDD, RDP, RCA, ICL, DIF, SLAD, DTE

**Relevance to Streaming Data Quality:**
- Contextual modeling addresses heterogeneous data
- Per-context thresholds for adaptive detection

**Relevance to CA-DQStream:**
- Conditional modeling framework for context-dependent quality scoring
- Bilevel optimization for context selection methodology
- Variance decomposition for quality metric design

---

### 10.5 Data Stream Clustering and Evolution

#### 10.5.1 Barddal et al. (2016) - SNCStream+

**Full Citation:**
Barddal, J.P., Gomes, H.M., Enembreck, F., & Barthès, J.-P. (2016). SNCStream+ : Extending a High Quality Anytime Data Stream Clustering Algorithm. *Information Systems*, 62, 60–73. https://doi.org/10.1016/j.is.2016.06.007

**BibTeX Entry:**
```bibtex
@article{barddal2016sncstream,
  title     = {SNCStream+: Extending a High Quality Anytime Data
               Stream Clustering Algorithm},
  author    = {Barddal, Jean Paul and Gomes, Heitor Murilo
               and Enembreck, Fabricio and Barthès, Jean-Paul},
  journal   = {Information Systems},
  volume    = {62},
  pages     = {60--73},
  year      = {2016},
  publisher = {Elsevier},
  doi       = {10.1016/j.is.2016.06.007}
}
```

**Method Proposed:**
- **Social Network Clusterer Stream+ (SNCStream+)**: True anytime clustering
- Models clustering as network formation/evolution problem
- Homophily-based edge rewiring for cluster formation
- Parameter ω controls network density/cluster count
- Optimizations: distance memoization, rewiring through dissipation

**Key Contributions:**
- No batch processing required (truly online)
- Automatic cluster count (no K parameter)
- Non-hyper-spherical cluster discovery
- Handles concept drift via exponential decay

**Datasets:**
| Dataset | Instances | Features | Domain |
|---------|----------|---------|--------|
| Airlines | 539,383 | 8 | Flight data |
| Electricity | 45,312 | 8 | Energy prices |
| Forest Covertype | 581,012 | 54 | Cartographic |
| KDD'99 | 4,898,431 | 42 | Intrusion detection |
| BPaM | 165,632 | 18 | Activity recognition |

**Results:**
- SNCStream+ achieves **0.97-0.99 CMM** on most datasets
- Superior clustering quality vs CluStream, ClusTree, DenStream, HAStream
- Fractional distance (L0.3) improves high-dimensional performance

**Relevance to Streaming Data Quality:**
- Cluster quality as data quality indicator
- Concept drift detection for quality monitoring

**Relevance to CA-DQStream:**
- Online clustering for quality pattern detection
- Adaptive window management for evolving data

---

### 10.6 Data Stream Pollution and Benchmarking

#### 10.6.1 Schinninger et al. (2025) - Icewafl Data Polluter

**Full Citation:**
Schinninger, C., Panse, F., Kühne, C., & Ehrlinger, L. (2025). Icewafl: A Configurable Data Stream Polluter. In *EDBT 2025* (pp. 796–802). OpenProceedings. https://doi.org/10.48786/edbt.2025.64

**BibTeX Entry:**
```bibtex
@inproceedings{schinninger2025icewafl,
  title     = {Icewafl: A Configurable Data Stream Polluter},
  author    = {Schinninger, Christoph and Panse, Fabian
               and Kühne, Constantin and Ehrlinger, Lisa},
  booktitle = {Proceedings of the 28th International Conference on
               Extending Database Technology (EDBT 2025)},
  pages     = {796--802},
  year      = {2025},
  publisher = {OpenProceedings},
  doi       = {10.48786/edbt.2025.64}
}
```

**Method Proposed:**
- **Icewafl (Inserting Customizable Errors with Apache Flink)**
- Novel temporal error types for streaming data
- Pollution pipelines with composable polluters
- Error conditions: random, value-dependent, temporal

**Error Taxonomy:**
| Category | Types | Temporal Aspect |
|----------|-------|----------------|
| Native | Delayed tuple, Frozen value, Timestamp error | Temporal by definition |
| Derived | Gaussian noise, Scaled by factor, Incorrect category, Missing value | Combined with change patterns |

**Key Features:**
- Event time as error function argument
- Composite polluters for complex scenarios
- Multi-stream integration for federated pollution
- Only 3-7% runtime overhead

**Datasets:**
- Beijing Air-Quality Dataset (420,768 tuples, 18 attributes, 4 years)
- Wearable Device Dataset (HR + activity data, 265 hours)

**Applications:**
1. DQ tool evaluation (Great Expectations)
2. Forecasting robustness testing (ARIMA, ARIMAX, Holt-Winters)

**Relevance to Streaming Data Quality:**
- Benchmark generation for DQ assessment
- Error injection methodology for testing quality detection

**Relevance to CA-DQStream:**
- Framework for generating ground-truth dirty streaming data
- Temporal error patterns to test quality detection sensitivity
- Integration with Apache Flink aligns with streaming infrastructure

---

### 10.7 Synthesis: Key Themes for CA-DQStream

#### 10.7.1 Common Threads Across Papers

| Theme | Papers | Relevance |
|-------|-------|----------|
| **Time-varying Quality** | Costa e Silva, Klein & Lehner, Restat | Need for temporal quality metrics that adapt |
| **Error Propagation** | Klein & Lehner, Restat | Operators affect quality dimensions differently |
| **Context Dependency** | King et al., Barddal | Quality varies by context/entity |
| **Real-time Requirements** | All papers | Low-latency processing essential |
| **Ground Truth** | Icewafl, Li et al. | Benchmark data for validation |

#### 10.7.2 Quality Dimensions Summary

| Dimension | Definition | Papers |
|----------|-----------|--------|
| **Accuracy** | Correctness of measured values | Klein & Lehner, Costa e Silva |
| **Completeness** | Absence of missing values | Klein & Lehner, Restat |
| **Timeliness** | Age/freshness of data | Klein & Lehner, Costa e Silva |
| **Consistency** | Uniformity across dataset | Costa e Silva |
| **Confidence** | Statistical error bounds | Klein & Lehner |

#### 10.7.3 Gaps and Opportunities

1. **Integrated Framework**: No paper combines all aspects (metrics + cleaning + detection + monitoring)
2. **Real-time Adaptation**: Dynamic quality threshold adjustment largely unexplored
3. **Cross-dimensional Dependencies**: How accuracy affects completeness perception not well studied
4. **Benchmark Standardization**: Need for unified streaming DQ benchmark

---

### Relevance to CA-DQStream

| Paper | Key Contribution | CA-DQStream Integration |
|-------|----------------|----------------------|
| Costa e Silva (2024) | Quality metrics (WQS, LWQS, QSD) | Core scoring framework |
| Klein & Lehner (2009) | DQ algebra for operators | Quality propagation through stream ops |
| Restat et al. (2025) | Streaming cleaning taxonomy | Error detection targets |
| Li et al. (2023) | Social media quality challenges | Domain-specific patterns |
| King et al. (2025) | Contextual anomaly detection | Conditional quality scoring |
| Barddal et al. (2016) | Online clustering | Quality pattern discovery |
| Icewafl (2025) | Benchmark generation | Ground truth testing |

---

## 11. Citation-Ready References (Streaming Data Quality)

### 11.1 Streaming Data Quality Metrics

**Costa e Silva, Oliveira, Oliveira (2024)**
```bibtex
@inproceedings{costa2024streaming,
  title     = {Enhancing Real-Time Analytics: Streaming Data Quality Metrics
               for Continuous Monitoring},
  author    = {Costa e Silva, Eliana and Oliveira, Óscar and Oliveira, Bruno},
  booktitle = {ICoMS 2024: 7th International Conference on Mathematics
               and Statistics},
  pages     = {97--101},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3686592.3686609}
}
```

**Klein & Lehner (2009)**
```bibtex
@article{klein2009representing,
  title     = {Representing Data Quality in Sensor Data Streaming Environments},
  author    = {Klein, Anja and Lehner, Wolfgang},
  journal   = {ACM Journal of Data and Information Quality},
  volume    = {1},
  number    = {2},
  pages     = {1--28},
  year      = {2009},
  publisher = {ACM},
  doi       = {10.1145/1577840.1577845}
}
```

### 11.2 Streaming Data Cleaning

**Restat et al. (2025)**
```bibtex
@article{restat2025cleaning,
  title     = {Data Cleaning of Data Streams},
  author    = {Restat, Valerie and Rodenhausen, Niklas
               and Antonin, Carina and Störl, Uta},
  journal   = {arXiv preprint arXiv:2507.20839},
  year      = {2025},
  eprint    = {2507.20839},
  archivePrefix = {arXiv},
  primaryClass = {cs.DB}
}
```

### 11.3 Streaming Anomaly Detection

**King et al. (2025)**
```bibtex
@article{king2025contextual,
  title     = {Contextual Learning for Anomaly Detection in Tabular Data},
  author    = {King, Spencer and Zhang, Zhilu and Yu, Ruofan
               and Coskun, Baris and Ding, Wei and Cui, Qian},
  journal   = {arXiv preprint arXiv:2509.09030},
  year      = {2025},
  eprint    = {2509.09030},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  version   = {v2}
}
```

### 11.4 Data Stream Clustering

**Barddal et al. (2016)**
```bibtex
@article{barddal2016sncstream,
  title     = {SNCStream+: Extending a High Quality Anytime Data
               Stream Clustering Algorithm},
  author    = {Barddal, Jean Paul and Gomes, Heitor Murilo
               and Enembreck, Fabricio and Barthès, Jean-Paul},
  journal   = {Information Systems},
  volume    = {62},
  pages     = {60--73},
  year      = {2016},
  publisher = {Elsevier},
  doi       = {10.1016/j.is.2016.06.007}
}
```

### 11.5 Streaming Benchmark Generation

**Schinninger et al. (2025)**
```bibtex
@inproceedings{schinninger2025icewafl,
  title     = {Icewafl: A Configurable Data Stream Polluter},
  author    = {Schinninger, Christoph and Panse, Fabian
               and Kühne, Constantin and Ehrlinger, Lisa},
  booktitle = {EDBT 2025: 28th International Conference on
               Extending Database Technology},
  pages     = {796--802},
  year      = {2025},
  publisher = {OpenProceedings},
  doi       = {10.48786/edbt.2025.64}
}
```

### 11.6 Social Media Event Detection

**Li et al. (2023)**
```bibtex
@article{li2023event,
  title     = {Event Detection from Social Media Stream: Methods,
               Datasets and Opportunities},
  author    = {Li, Quanzhi and Chao, Yang and Li, Dong
               and Lu, Yao and Zhang, Chi},
  journal   = {IEEE Access},
  pages     = {1--22},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2023.0000}
}
```

---

## 11b. Survey Papers: Real-Time Event Detection

This section covers two foundational survey papers: (1) a comprehensive survey on Twitter-based real-time event detection, and (2) a literature review on data quality processing in streaming environments.

---

### 11b.1 Hasan et al. (2018): Survey on Real-Time Event Detection from Twitter

#### Full Citation

**APA Citation:**
Hasan, M., Orgun, M. A., & Schwitter, R. (2018). A survey on real-time event detection from the Twitter data stream. *Journal of Information Science*, 44(4), 443–463. https://doi.org/10.1177/0165551517698564

**BibTeX Entry:**
```bibtex
@article{hasan2018survey,
  title     = {A survey on real-time event detection from the
               Twitter data stream},
  author    = {Hasan, Mahmud and Orgun, Mehmet A. and Schwitter, Rolf},
  journal   = {Journal of Information Science},
  volume    = {44},
  number    = {4},
  pages     = {443--463},
  year      = {2018},
  publisher = {SAGE Publications},
  doi       = {10.1177/0165551517698564}
}
```

#### Method Proposed

This paper provides a comprehensive survey of real-time event detection methods applied to streaming Twitter data, classifying them into four main categories:

1. **Term-Interestingness-Based Approaches:** Detect events by tracking bursty terms (keywords/phrases) that spike in frequency. Methods include:
   - **Twevent** (Li et al.): Identifies top-k bursty word segments using Microsoft Web N-gram + Wikipedia; newsworthiness score computed from Wikipedia anchor text priors
   - **TwitterMonitor** (Mathioudakis & Koudas): Elementary queuing model for high-frequency term detection; greedy grouping strategy
   - **EDCoW** (Weng & Lee): Discrete wavelet signals for filtering trivial words; modularity-based graph partitioning for clustering
   - **SAX*** (Stilo & Velardi): Symbolic Aggregate ApproXimation (SAX) for temporal series discretization; Wikipedia Events regex for non-event term removal
   - **MABED** (Guille & Favre): Mention-Anomaly-Based Event Detection — exploits textual content and user-mention frequency; characterizes events by duration, main word, and magnitude

2. **Topic-Modelling-Based Approaches:** Use probabilistic latent topic models (LDA, HDP, etc.) to discover hidden semantic structures:
   - **TwiCal:** Conditional Random Fields for event phrase extraction; G2 log-likelihood for entity-date association ranking
   - **LECM** (Zhou et al.): Joint distribution over named entities, date/time, location, and terms; Freebase API for semantic mapping
   - **GEAM:** Hierarchical Bayesian model distinguishing event-related aspects from general topics via switching variable
   - **TopicSketch** (Xie et al.): Sketch-based acceleration detection on stream, word, and word-pair quantities; hashing-based dimension reduction; lazy maintenance O(H·d²)
   - **BEE+** (Li et al.): Incremental topic model with distributed execution; single-topic-per-microblog assumption; faster convergence than standard LDA

3. **Incremental-Clustering-Based Approaches:** Similarity-threshold-based clustering with unknown cluster count; handles streaming data's dynamic nature:
   - **TwitterNews+** (Hasan et al.): Two-stage (Search Module + EventCluster Module); inverted index for O(1) similarity decision
   - **First Story Detection** (Petrovic et al.): LSH-based incremental clustering; cluster expiration after fixed time; entropy-based ranking
   - **Becker et al.:** SVM classifier for event vs. non-event distinction; confidence score for ranking
   - **McMinn & Jose:** Named entity burst detection; Three Sigma rule (3σ) for spike detection; entity-specific inverted indexes
   - **LSED** (Unankard et al.): Leader–follower clustering with content + concept similarity; location correlation scoring

4. **Miscellaneous Approaches:** Hybrid techniques including linear classifiers on user communication behavior (Chierichetti et al.), High Utility Pattern mining (Huang et al.), multi-view topic detection (Fang et al.), and disease outbreak detection (Thapen et al., Sakaki et al. earthquake prediction with particle filters).

#### Key Results / Metrics

| Method | Precision | Recall | DERate |
|--------|-----------|--------|--------|
| Twevent | 0.861 | 75/101 events | 0.16 |
| TwitInfo | 0.80–0.95 | 0.80–0.95 | — |
| TLDF | 0.923 | — | — |
| SAX* | 0.79–0.91 | — | — |
| LSED | 0.973 | 90/136 | — |
| De Boom et al. | 0.64 | 0.377 | — |
| McMinn & Jose | 0.636 | 194/506 | — |
| BEE+ | 0.70 | 0.733 | — |
| ET (Parikh) | 0.91 | 21/23 | — |
| MABED | 0.775 | 0.608 | 0.167 |

#### Dataset Used

- **Twevent evaluation:** 4.3 million tweets from Singapore-based users (1 month)
- **TwitInfo evaluation:** Soccer game tweets + earthquake tweets (manual ground truth)
- **TLDF evaluation:** FIFA World Cup 2014 (64 matches)
- **SAX* evaluation:** 1% of 1-year Twitter stream
- **LSED evaluation:** ~200,000 tweets over 1 week
- **McMinn & Jose / Events2012 corpus:** 120 million tweets with relevance judgments for 506 events (publicly available ground truth corpus)
- **Zhou et al. (LECM):** 64 million tweets (December 2010)
- **BEE+ evaluation:** Sina Weibo (Chinese microblog) corpus
- **MABED evaluation:** ~1.4 million English tweets (November 2009)

#### Comparison with Baselines

- **Twevent vs. EDCoW baseline:** +9.9% precision improvement (0.861 vs. 0.762); recall 75/101 vs. 13/21
- **TLDF vs. TwitterMonitor + enBlogue:** +23.6–37.7% precision improvement over TwitterMonitor; +15.8–23.6% over enBlogue
- **LSED vs. Sayyadi, Ozdikis, enBlogue, LEED baselines:** Best precision at 0.973; best recall at 90/136
- **McMinn & Jose vs. Aggarwal & LSH (Petrovic):** +35.1% precision improvement (0.636 vs. 0.048–0.285); +38–162 more events detected
- **BEE+ vs. TwitterMonitor, PLSA, BEE:** +20% precision over 0.50 baselines; competitive recall at 0.733
- **MABED vs. TS, ET, α-MABED:** +15–20% precision improvement; best overall F1

#### Key Observations and Challenges

- **Challenges:** Limited context (140-character limit), high noise ratio, informal language, spelling errors, spam
- **Fragmentation problem:** Incremental clustering prone to detecting same event multiple times
- **Threshold sensitivity:** Similarity thresholds must adapt to dynamic topic changes
- **Data quality issues:** Tweet credibility, spam filtering, and retweet deduplication affect detection quality
- **Evaluation gaps:** Lack of publicly available labeled corpora; Events2012 (McMinn et al.) is the notable exception
- **Latency advantage:** Twitter leads Facebook/Google+ in breaking news; comparable to newswire for major events; better coverage for sports, unpredictable events, and local events

---

### 11b.2 Benabbas & Nicklas (2024): Data Quality Processing in Streaming Environments

#### Full Citation

**APA Citation:**
Benabbas, A., & Nicklas, D. (2024). Data Quality Processing in Data Streaming Environments: A Literature Review. University of Bamberg. https://doi.org/10.20378/irb-112076

**BibTeX Entry:**
```bibtex
@article{benabbas2024dqstreaming,
  title  = {Data Quality Processing in Data Streaming Environments:
            A Literature Review},
  author = {Benabbas, Aboubakr and Nicklas, Daniela},
  year   = {2024},
  institution = {University of Bamberg},
  note   = {https://doi.org/10.20378/irb-112076},
  url    = {https://doi.org/10.20378/irb-112076}
}
```

#### Method Proposed

This literature review provides a structured overview of data quality (DQ) concepts, representations, and processing techniques for continuous data streams and sensor-driven systems. Key contributions:

**DQ Representation:**
- **DQ Models & Standards:** ISO/IEC 25012 for data quality characteristics categorization
- **Quality-Aware Query Processing:** Ranking/filtering records based on quality scores; quality metadata embedded in query optimization
- **Adaptive Quality Representation:** Adaptive Timeliness (tighter freshness during high-risk periods); Adaptive Completeness (relaxed for exploratory analytics, strict for regulatory reporting)
- **DQ Labeling:** Quality indicators (accuracy, timeliness, confidence) attached to individual data elements as annotations
- **Error Propagation Estimation:** Models how errors propagate through transformations and aggregations

**DQ Dimensions (Streaming-Oriented):**

| Dimension | Streaming Definition | Representative Metrics |
|-----------|---------------------|----------------------|
| Accuracy | Degree to which value reflects real-world phenomenon (sensor noise, calibration) | Error deviation, RMSE, proportion of validated observations |
| Completeness | Extent to which expected data points/attributes are present (packet loss, offline sensors) | Missing-value ratio, record availability ratio, attribute coverage |
| Consistency | Conformance to physical/logical/semantic constraints within and across streams | Rule violations, unit-conversion conflicts, cross-stream agreement ratio |
| Timeliness | Degree to which data arrives within its validity window | Data age distribution, lateness ratio, freshness indicator |
| Traceability | Ability to track origin and transformation sequence | Provenance completeness index, lineage depth |
| Duplication | Redundant tuples due to retransmissions, buffering | Duplicate-tuple ratio, deduplication effectiveness |
| Volatility | Frequency/magnitude of measurement changes over time | Update rate, temporal variance index, change-frequency ratio |

**DQ Processing Methods:**
- **Sensor data cleaning:** Hybrid declarative rules + statistical modeling (Gill & Lee, 2015)
- **Outlier detection:** Extensive WSN outlier detection survey (Ayadi et al., 2017)
- **Extensible frameworks:** ESP (Jeffery et al., 2006) — programmable architecture for real-time sensor data cleaning
- **ML/AI approaches:** Deep learning for adaptive anomaly detection; federated learning for distributed privacy-preserving quality processing
- **Pattern-based approaches:** Data Quality Patterns for reusable, adaptable quality strategies

#### Key Results / Metrics

The paper synthesizes rather than evaluates; representative metrics from the literature:

| Metric Category | Description |
|----------------|-------------|
| Content-based metrics | Intrinsic data properties (accuracy, completeness, consistency) — most relevant for streaming at raw-data level |
| Query-based metrics | Precision, recall, relevance — depend on user intent |
| Application-based metrics | Task-specific thresholds and performance expectations |
| ISO/IEC 25012 compliance | Standardized data quality characteristic assessment |

#### Dataset Used

Literature review synthesizing findings from multiple sources:
- Sensor networks and WSN (Wireless Sensor Networks) data streams
- IoT and environmental monitoring data
- Financial trading and autonomous vehicle data contexts
- General big data and streaming environments

#### Comparison with Baselines

- Systematic review methodology (Teh et al., 2020) for sensor-related data errors
- Comparison of DQ models: Wang & Strong (1996), Batini et al. (2009), ISO/IEC 25012 (2008)
- Geisler et al. (2016) categorization: content-based, query-based, and application-based metrics
- Foundational works: Shannon (1948) error detection/correction, Codd (1970) relational model, Kimball & Ross (1996) data warehousing

#### Key Observations and Research Gaps

1. **Domain specificity vs. generality:** Most DQ processing solutions are tailored to specific use cases, limiting cross-domain applicability
2. **Expert dependency:** Integrating quality assessment into pipelines remains challenging for non-expert users
3. **Adaptive requirements:** Streaming environments require adaptive, context-aware approaches — static batch-era frameworks are insufficient
4. **Volume/velocity/variability challenges:** High-velocity continuous data cannot be easily revisited once processed
5. **AI-driven promise:** Deep learning for automatic quality assessment; federated learning for distributed quality processing
6. **Pattern-based frameworks:** Data Quality Patterns narrowing task focus while ensuring simplicity and adaptability
7. **Usability gap:** Substantial technical expertise required for most existing tools; need for intuitive interfaces and automated routine tasks

---

### Relevance to CA-DQStream

| Aspect | Hasan et al. (2018) Survey | Benabbas & Nicklas (2024) Review |
|--------|---------------------------|-----------------------------------|
| **Streaming data quality** | Twitter's noise, informal language, and spam directly mirror streaming DQ challenges (missing values, outliers, inconsistency) | Core focus — provides the foundational DQ dimension framework (accuracy, completeness, consistency, timeliness) for CA-DQStream |
| **Adaptive thresholds** | Similarity thresholds must adapt to dynamic Twitter topics; fragmentation problem mirrors quality score drift | Adaptive quality representation directly applicable — Adaptive Timeliness and Adaptive Completeness inform CA-DQStream's dynamic thresholding |
| **Incremental processing** | Incremental clustering essential for unknown cluster counts in streaming; similar to CA-DQStream's need for incremental quality profile updates | Real-time DQ assessment requires continuous, incremental processing as data arrives |
| **Context-awareness** | Location-based event detection (LSED), temporal burst patterns, named entity context inform geographic and temporal quality profiling | DQ cannot be meaningfully represented without context; requirements vary across domains — directly motivates CA-DQStream's 5W1H framework |
| **Evaluation methodology** | Events2012 corpus (120M tweets, 506 events) as gold standard — parallels need for labeled streaming DQ benchmark datasets | Content/query/application-based metric categorization guides CA-DQStream's multi-dimensional quality scoring |
| **Processing pipeline** | Pre-processing (POS tagging, NER, slang conversion, tweet filtering >90% removal) parallels CA-DQStream's data validation and cleaning stages | DQ processing taxonomy (fault detection, data cleaning, anomaly management) maps to CA-DQStream's dual-branch architecture |
| **Scalability challenge** | High-volume Twitter stream requires constant-time/space algorithms; EDCoW's wavelet approach computationally expensive | High-velocity streaming demands efficient algorithms; Benabbas & Nicklas highlight need for scalable, autonomous solutions |
| **Data quality dimensions** | Tweet credibility, spam filtering, and retweet deduplication as quality signals | Six core DQ dimensions (accuracy, completeness, consistency, timeliness, traceability, duplication) form the basis for CA-DQStream's multi-dimensional quality scoring model |
| **Gap addressed** | Lack of labeled ground truth; most evaluations rely on manual inspection | DQ solutions are domain-specific and require expert intervention; lack flexibility — CA-DQStream's domain-independent adaptive framework addresses both |
| **Future directions** | Combining social, temporal, and topical features for better event ranking; tweet credibility detection for quality filtering | AI-driven methods, pattern-based frameworks, semantic models — CA-DQStream's ML-based quality scorer aligns with these emerging trends |

*Document generated: Streaming Data Quality Academic Papers Summary*
*Total papers reviewed: 8*
*Last updated: May 2026*

---

## 2c. Kafka & Flink for Streaming Data Quality (Academic + Industry)

This section surveys academic papers and industry engineering resources on data quality monitoring systems built on Apache Kafka and Apache Flink — the dominant open-source stack for streaming data pipelines. It covers production-scale validation frameworks, schema enforcement, anomaly detection built into Flink SQL, and monitoring architectures. These sources collectively establish the engineering substrate upon which CA-DQStream operates and the production pain points it aims to address.

---

### 2c.1 Real-Time Data Quality Monitoring: Kafka Stream Contracts with Syntactic and Semantic Tests

**Source:** Grab Engineering Blog (Grab is a leading Southeast Asian superapp; FlinkSQL-based platform deployed in production)

**URL:** https://engineering.grab.com/real-time-data-quality-monitoring

#### Method / Approach

The **Coban Platform** at Grab implements a full data-quality-as-a-service framework for Kafka streams consisting of four components:

1. **Data Contract Definition** — Stakeholders define formal contracts specifying schema agreements, semantic field-level rules (e.g., string pattern, number range, constant value), and ownership details for alerting. Contracts include both syntactic (schema-level) and semantic (field-level business logic) rules.
2. **LLM-Assisted Rule Recommendation** — Recognizing that defining hundreds of field-specific rules is cognitively burdensome, the platform uses large language models to predict semantic test rules from provided Kafka stream schemas and anonymized sample data, dramatically reducing setup friction.
3. **Automated Test Execution via FlinkSQL** — A long-running **Test Runner** consumes Kafka topic data using its own consumer group (isolated from production consumers) and executes inverse-SQL queries derived from the data contract to capture any record violating syntactic or semantic rules. FlinkSQL was chosen specifically for its flexibility in expressing test rules as standard SQL.
4. **Result Observability** — Violating records are published to a dedicated Kafka topic and simultaneously written to AWS S3. An in-house observability platform (Genchi) consumes these events, surfaces them in a UI showing which fields and values violated which rules, and sends Slack notifications to stream owners with links to sample bad records, observed time windows, and bad-record counts.

#### Key Results / Metrics

- Deployed and actively monitoring **100+ critical Kafka topics** across Grab's production environment.
- Enables immediate identification and halting of invalid data propagation across multiple streams.
- Shift from reactive batch correction (hours/days to detect) to **proactive prevention** with near-real-time alerts.

#### Architecture Components

- **Kafka** as the streaming substrate and dead-letter topic destination.
- **FlinkSQL Test Runner** as the compute engine for real-time rule evaluation.
- **Coban UI** for contract definition, rule authoring, and result visualization.
- **LLM recommendation engine** for semantic rule generation.
- **Genchi observability platform** for event consumption and Slack alerting.
- **AWS S3** for deep observability and historical bad-record storage.

#### Comparison with Baselines

Prior to Coban, Grab's Kafka stream quality monitoring relied on manual checks with no automated real-time validation. The Coban platform represents the first automated, systematic solution — enabling declarative contracts, continuous execution, and owner-level alerting at the stream level. This is qualitatively distinct from generic schema registry validation (which handles only syntactic compatibility) and batch-based DQ checks (which introduce hours of detection latency).

---

### 2c.2 Making Data Quality Scalable with Real-Time Streaming Architectures

**Source:** Confluent Blog (Confluent is the primary commercial sponsor and Kubernetes-scale provider of Apache Kafka; authors are Kafka/stream processing engineers)

**URL:** https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/

#### Method / Approach

This article establishes the foundational shift-left principle for streaming data quality: **validation must be built into the pipeline at or near the source**, not applied post-hoc in batch jobs. The authors describe a two-layer continuous quality framework:

1. **Validation Layer** — Real-time checks applied at ingestion, including schema enforcement via Confluent Schema Registry (BACKWARD/FORWARD/FULL compatibility modes), field-level type and range validation, and business-rule enforcement using Common Expression Language (CEL) data contracts. This layer prevents bad data from entering the pipeline.
2. **Monitoring Layer** — Continuous monitoring of the six DQ dimensions (accuracy, completeness, consistency, timeliness, validity, uniqueness) as data flows through Kafka topics and Flink processing jobs. Alerts fire when dimension metrics breach pre-defined thresholds.

The article also documents a Siemens Healthineers case study: a production pipeline processing **8 million messages per day** that moved from daily batch quality checks to continuous streaming validation, catching schema violations and semantic anomalies within seconds of occurrence.

#### Key Results / Metrics

- Siemens Healthineers: **8 million messages/day** processed through continuous quality validation.
- Detection latency reduced from **hours/days (batch) to seconds (streaming)**.
- Shift-left approach prevents bad data from propagating downstream to ML models and analytics dashboards.

#### Architecture Components

- **Confluent Schema Registry** for Avro/Protobuf/JSON Schema enforcement.
- **CEL (Common Expression Language) data contracts** for declarative field-level validation rules.
- **Dead Letter Queues (DLQ)** for routing invalid records without blocking main processing.
- **Apache Kafka** as the streaming backbone.
- **Kafka Connect** with Single Message Transforms (SMTs) for inline validation.
- **Confluent Cloud** managed Flink for continuous monitoring.

#### Comparison with Baselines

The article explicitly contrasts with traditional batch validation: batch DQ checks (run hourly or daily) are fundamentally too late — by the time a batch failure is detected, bad data has already contaminated downstream dashboards, ML training sets, and business decisions. The Confluent approach and Grab's Coban platform both share the shift-left philosophy but differ in implementation: Confluent emphasizes schema-registry-native enforcement at the broker layer, while Coban uses FlinkSQL-based consumer-side contracts with richer semantic rule support.

---

### 2c.3 Detect Anomalies in Data with Confluent Cloud for Apache Flink (AI-Based Built-in Functions)

**Source:** Confluent Documentation (Confluent Cloud, built-in AI functions; published 2025)

**URL:** https://docs.confluent.io/cloud/current/ai/builtin-functions/detect-anomalies.html

#### Method / Approach

Confluent Cloud for Apache Flink exposes **built-in anomaly detection functions** as first-class SQL primitives, enabling streaming ML without custom model deployment:

1. **`AI_DETECT_ANOMALIES`** — Uses **Google's TimesFM** (Time-Series Foundation Model) to forecast expected values for a metric stream and flag deviations from the forecast. This is a production-ready foundation model approach to anomaly detection within Flink SQL.
2. **`ML_DETECT_ANOMALIES`** — Statistical and ML-based detection using **ARIMA models** for time-series forecasting, with configurable sensitivity thresholds. Suitable for structured metric streams where historical patterns inform expected behavior.

Both functions are invoked as SQL expressions within Flink SQL queries, eliminating the need for custom Python/Java UDFs or external ML serving infrastructure.

#### Key Results / Metrics

- Foundation model (TimesFM) provides zero-shot forecasting — no training data required per metric stream.
- ARIMA-based approach requires minimal hyperparameter tuning and integrates natively with Flink's windowing and watermark semantics.
- Both functions support **exactly-once processing** semantics, ensuring anomaly labels are consistent with the underlying data.

#### Architecture Components

- **Confluent Cloud for Apache Flink** — Managed Flink as a service with built-in AI function registry.
- **Google TimesFM foundation model** — Pre-trained time-series forecasting model, invoked via `AI_DETECT_ANOMALIES`.
- **ARIMA** — Classical statistical forecasting invoked via `ML_DETECT_ANOMALIES`.
- **Flink SQL** — User-facing API for composing streaming queries with anomaly detection.
- **Confluent Schema Registry** — Ensures typed, validated data entering the anomaly detection pipeline.

#### Comparison with Baselines

Traditional anomaly detection in streaming pipelines requires: (a) selecting and training a model, (b) deploying it as a serving endpoint, (c) building a connector from Flink to the endpoint. Confluent's built-in functions eliminate all three steps — the model is pre-trained (TimesFM) or auto-parameterized (ARIMA), hosted within the Flink runtime, and callable as a SQL function. This is qualitatively different from the rule-based approaches in Grab's Coban platform (syntactic/semantic contracts) and represents the ML-based scoring branch of the dual-branch architecture.

---

### 2c.4 AI-Governed Multi-Modal Data Sourcing Pipelines Using Apache Flink on Kubernetes

**Source:** Computer Fraud and Security (peer-reviewed journal, Volume 2026, Issue 1; published March 10, 2026)

**URL:** https://computerfraudsecurity.com/index.php/journal/article/view/966

#### Method / Approach

This recent journal article proposes a comprehensive AI-governed framework for multi-modal data sourcing in cloud lakehouse environments, with three core innovations relevant to streaming data quality:

1. **Semantic Contracts** — Replace rigid schemas with machine-learned expectations encoding attribute relationships, value distributions, and contextual meanings derived from historical patterns. Unlike schema registries that enforce exact types and formats, semantic contracts learn what values are *normal* for each attribute over time and flag deviations — a fundamentally richer quality signal than binary schema conformance.
2. **Self-Evolving Schema Intelligence** — Uses **large language models and embedding-based similarity scoring** to detect structural schema drift (new fields, renamed fields, type changes), infer field-level transformations, and autonomously generate adaptation logic without manual intervention.
3. **Cross-Format Normalization** — AI-based extraction engines handle unstructured text, recursive parsing for semi-structured hierarchies (JSON, Avro), and temporal alignment mechanisms for event ordering across sources with varying latency — directly relevant to the timeliness dimension of data quality.
4. **Apache Flink on Kubernetes** — Distributed stream processing with elastic scaling, stateful computation, and exactly-once semantics. Asynchronous barrier snapshots enable lightweight checkpointing without halting stream execution. Kubernetes provides automated resource allocation, failure recovery, and operator lifecycle management.

#### Key Results / Metrics

- Asynchronous barrier snapshots enable lightweight checkpointing without halting stream execution.
- Autonomous adaptation capabilities documented for schema drift, semantic inconsistency, and format heterogeneity.
- Operational resilience for rapidly evolving lakehouse deployments with multi-source data ingestion.

#### Architecture Components

- **Semantic contracts** — Learned expectations per attribute (value distributions, relationships).
- **LLM + embedding-based schema intelligence** — Drift detection, transformation inference, adaptation logic generation.
- **Cross-format normalization engine** — Unstructured text extraction, semi-structured parsing, temporal alignment.
- **Apache Flink** — Stream processing engine with exactly-once semantics.
- **Kubernetes** — Container orchestration for elastic scaling and failure recovery.
- **Cloud lakehouse** — Delta Lake / Apache Iceberg destination for processed data.

#### Comparison with Baselines

This approach is distinguished from conventional schema registries (Confluent), rule-based DQ frameworks (Coban), and built-in ML functions (Confluent AI_DETECT_ANOMALIES) by its **self-evolving, learning-based nature**. Schema registries require manual evolution planning; Coban requires human-authored semantic rules; Confluent's AI functions require pre-defined metric streams. The AI-governed approach autonomously learns expected data behavior, detects drift without predefined thresholds, and generates adaptation logic — representing the most automated and context-aware of the systems surveyed.

---

### 2c.5 Flink Job Monitoring: Key Metrics and Alerting Strategies for Production

**Source:** Streamkap Blog (Streamkap is a managed CDC and streaming data platform; published February 25, 2026)

**URL:** https://streamkap.com/resources-and-guides/flink-job-monitoring-metrics

#### Method / Approach

This production operations guide establishes the five essential metric categories for monitoring Flink-based streaming pipelines, with specific alerting strategies:

1. **Throughput Metrics** — `numRecordsInPerSecond`, `numRecordsOutPerSecond`, byte-level throughput. Monitored at the operator level (not just job level) to detect per-partition skew. Compare against 7-day rolling baselines rather than static thresholds.
2. **Checkpoint Metrics** — `lastCheckpointDuration`, `lastCheckpointSize`, `numberOfFailedCheckpoints`, alignment duration. Checkpoint duration is a **leading indicator** — a 5s → 30s trend predicts imminent failure before the actual failure occurs.
3. **State Metrics** — RocksDB state backend metrics (`estimate-num-keys`, `estimate-live-data-size`, `compaction-pending`). Unbounded state growth is a primary cause of Flink job degradation.
4. **Backpressure Metrics** — `isBackPressured`, `busyTimeMsPerSecond`, `backPressuredTimeMsPerSecond`. Backpressure propagates *upstream* — the bottleneck operator is the first *non*-backpressured operator with high busy time.
5. **Consumer Lag** — `records-lag-max` for Kafka-sourced pipelines. Lag growing continuously (even if below a threshold) is the critical alert trigger.

Alert strategy emphasizes **trends over thresholds**: use `deriv()` and `avg_over_time()` in PromQL to detect gradual degradation rather than instantaneous spikes. The guide provides a complete Prometheus scrape configuration and a recommended 4-row Grafana dashboard template for CDC pipelines.

#### Key Results / Metrics

- **Prometheus + Grafana** integration via Flink's built-in metrics reporter.
- Alert on trends (checkpoint duration growing 30 minutes) not spikes.
- Consumer lag growth rate (records/minute) as the primary early warning signal.
- RocksDB compaction pending as a predictor of read performance degradation.
- Operational runbooks provided for: job failure, checkpoint failures, consumer lag, backpressure, state size growth.

#### Architecture Components

- **Apache Flink** metrics reporter (Prometheus exporter on port 9249).
- **Prometheus** scrape configuration for Flink TaskManagers via Kubernetes pod annotations or static targets.
- **Grafana** dashboard (community template ID 14911) customized with 4-row CDC pipeline layout.
- **Alertmanager** inhibition rules to suppress downstream symptom alerts when root cause is already firing.
- **Kubernetes** pod annotations for automated service discovery.

#### Comparison with Baselines

This operational guide fills the gap between Flink's rich metrics surface and actionable production monitoring. The approach is distinct from schema-level validation (Confluent), semantic contracts (Coban), and ML-based anomaly detection (Confluent AI functions) — it addresses **pipeline health monitoring** as a prerequisite for data quality. CA-DQStream's quality scores are directly affected by all five metric categories: throughput drops cause missed records (completeness), checkpoint failures cause data duplication/inconsistency, consumer lag directly impacts timeliness, and backpressure can cause out-of-order event processing.

---

### 2c.6 StreamDQ: Data Quality Monitoring for Apache Flink

**Source:** GitHub Repository — stefan-grafberger/StreamDQ (https://github.com/stefan-grafberger/StreamDQ)

#### Method / Approach

StreamDQ is a Kotlin-based library built specifically on Apache Flink for **continuous data quality measurement** in streaming pipelines. It implements the concept of "unit tests for data" — enabling developers to define quality checks that run continuously as data flows through Flink jobs. Supports both row-level checks (individual record validation) and aggregate checks (statistical properties over sliding windows).

#### Key Results / Metrics

- Integrates directly with Flink's DataStream and Table API.
- Supports the six DQ dimensions: completeness, timeliness, accuracy, consistency, uniqueness, validity.
- Designed for large-scale stream processing with Flink's distributed execution model.

#### Architecture Components

- **Apache Flink** as the processing engine.
- **Kotlin** as the implementation language.
- **Row-level checks** — individual record validation (null checks, type checks, range checks).
- **Aggregate checks** — windowed statistical assertions (distribution checks, freshness checks).

#### Comparison with Baselines

StreamDQ targets the developer experience angle — making DQ testing as natural as unit testing. Unlike the platform-level solutions (Coban, Confluent), StreamDQ is a library that developers embed directly in Flink jobs, giving fine-grained control over which operators perform which checks. This contrasts with consumer-side approaches (Coban's isolated consumer group) and broker-side approaches (Confluent Schema Registry). StreamDQ represents the in-processing quality enforcement model.

---

### Full Citations

**Grab Engineering. (2024).** Real-time data quality monitoring: Kafka stream contracts with syntactic and semantic tests. *Grab Engineering Blog*. https://engineering.grab.com/real-time-data-quality-monitoring

**Confluent. (2024).** Making data quality scalable with real-time streaming architectures. *Confluent Blog*. https://www.confluent.io/blog/making-data-quality-scalable-with-real-time-streaming-architectures/

**Confluent. (2025).** Detect anomalies in data with Confluent Cloud for Apache Flink. *Confluent Documentation*. https://docs.confluent.io/cloud/current/ai/builtin-functions/detect-anomalies.html

**Computer Fraud and Security. (2026).** AI-governed multi-modal data sourcing pipelines using Apache Flink on Kubernetes: A self-evolving architecture for semantic contracts, schema intelligence, and cross-format normalization in cloud lakehouse systems. *Computer Fraud and Security*, 2026(1). https://computerfraudsecurity.com/index.php/journal/article/view/966

**Streamkap. (2026).** Flink job monitoring: Key metrics and alerting strategies. *Streamkap Blog*. https://streamkap.com/resources-and-guides/flink-job-monitoring-metrics

**Grafberger, S. (n.d.).** StreamDQ: Data quality monitoring for Apache Flink. GitHub repository. https://github.com/stefan-grafberger/StreamDQ

**Streamkap. (n.d.).** Data quality in streaming pipelines: A practical framework. *Streamkap*. https://streamkap.com/resources-and-guides/data-quality-streaming-pipelines

---

### Relevance to CA-DQStream

| Aspect | Relevance |
|--------|-----------|
| **Kafka + Flink substrate** | All six sources use Kafka as the streaming backbone and Flink as the processing/compute engine. CA-DQStream inherits this substrate. |
| **Six DQ dimensions** | Confluent, Streamkap, Grab Coban, and StreamDQ all reference completeness, timeliness, accuracy, consistency, validity, and uniqueness as the canonical quality dimensions — CA-DQStream's quality score taxonomy aligns directly. |
| **FlinkSQL for DQ validation** | Grab's Coban platform uses FlinkSQL as the test execution engine, validating that FlinkSQL is production-viable for DQ workloads at scale (100+ topics, 8M messages/day). CA-DQStream's dual-branch architecture (rule engine + ML scorer) can leverage FlinkSQL for the rule-evaluation branch. |
| **LLM-based semantic rules** | Coban's LLM-assisted rule recommendation and the AI-governed paper's self-evolving schema intelligence both point to LLMs reducing human burden in authoring quality rules. CA-DQStream can incorporate LLM-assisted rule generation as a future enhancement. |
| **Shift-left validation** | Both Confluent and Grab advocate building validation into pipelines at/near the source rather than post-hoc. CA-DQStream's scoring architecture supports this by enabling per-operator quality checkpoints. |
| **Dead Letter Queues** | Confluent and Streamkap emphasize DLQ routing for graceful degradation. CA-DQStream should integrate DLQ routing for invalid records flagged by the rule-evaluation branch. |
| **Foundation model anomaly detection** | Confluent's `AI_DETECT_ANOMALIES` (TimesFM) and the AI-governed paper's LLM + embedding approach demonstrate that pre-trained/foundation models are production-ready for streaming anomaly detection. CA-DQStream's ML scoring branch aligns with this trend. |
| **Self-evolving schema intelligence** | The AI-governed paper's approach of learning expected attribute relationships over time and autonomously detecting drift maps directly to CA-DQStream's context-profile learning — where zone-aware and periodicity-aware profiles evolve from historical data. |
| **Flink metrics as quality signals** | Streamkap's five essential Flink metrics (throughput, checkpoints, state, backpressure, consumer lag) provide the operational health layer beneath quality scores. CA-DQStream's quality scores should be cross-referenced with Flink operational metrics for root-cause analysis. |
| **Consumer lag as timeliness proxy** | Streamkap identifies consumer lag as the primary end-to-end health metric for Kafka-sourced pipelines. Consumer lag maps directly to CA-DQStream's timeliness quality dimension — bridging operational monitoring and data quality scoring. |

---

*Document generated: Streaming Data Quality Academic Papers Summary*
*Total papers reviewed: 8*
*Last updated: May 2026*

