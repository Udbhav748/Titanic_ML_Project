# Data Dictionary

Every column after Stage 1 cleaning — raw columns plus everything `TitanicPreprocessor` adds. Same source as `artifacts/feature_schema.json`.

| Feature | Type | Source | Description | Used in Production |
|---|---|---|---|---|
| Survived | Binary | Original dataset | Target label: 1 = survived, 0 = did not survive. | Target (label) |
| PassengerId | Numeric (identifier) | Original dataset | Unique row identifier; no predictive signal. | Deprecated |
| Pclass | Categorical (ordinal) | Original dataset | Ticket class (1/2/3); proxy for socioeconomic status and physical deck location. | Yes |
| Name | Text | Original dataset | Free-text passenger name; not used directly, but the source column for Title extraction. | Deprecated |
| Sex | Categorical | Original dataset | Passenger sex. | Yes |
| Age | Numeric | Original dataset (imputed) | Age in years. Nullable on input; TitanicPreprocessor fills missing values with the fitted per-Title training median, never a static default. | Yes |
| SibSp | Numeric (count) | Original dataset | Siblings/spouses aboard. | Yes |
| Parch | Numeric (count) | Original dataset | Parents/children aboard. | Yes |
| Ticket | Text | Original dataset | Ticket number; high-cardinality identifier with no clean signal. | Deprecated |
| Fare | Numeric | Original dataset | Fare paid; right-skewed (flagged for a log-transform check in Stage 2). | Yes |
| Cabin | Text | Original dataset | 77% missing. Not used directly — decomposed into HasCabin and Deck instead. | Deprecated |
| Embarked | Categorical | Original dataset (imputed) | Port of embarkation. Nullable on input; TitanicPreprocessor fills missing values with the fitted training-split mode (currently 'S'). | Yes |
| HasCabin | Binary | Engineered (from Cabin) | 1 if a cabin was recorded, 0 otherwise. | Yes |
| FamilySize | Numeric (count) | Engineered (SibSp + Parch + 1) | Total family members aboard including the passenger; non-monotonic effect on survival (see reports/stage1_data_audit.md). | Yes |
| IsAlone | Binary | Engineered (FamilySize == 1) | 1 if traveling without any family aboard. Deterministic function of FamilySize. | Yes |
| Title | Categorical | Engineered (from Name) | Honorific extracted from Name; titles occurring <10 times in the training split (or never seen at fit time) collapse to 'Rare'. | Yes |
| FarePerPerson | Numeric | Engineered (Fare / FamilySize) | Estimated per-individual fare, adjusting for group ticket pricing. Divide-by-zero guarded. | Candidate |
| Deck | Categorical | Engineered (from Cabin) | First letter of Cabin; 'U' where Cabin is missing, 'Other' where an unseen deck letter appears at inference. Highly redundant with HasCabin. | Candidate |
| AgeGroup | Categorical | Engineered (binned Age) | Age binned into Child (<=12) / Teen (13-18) / Adult (19-59) / Senior (60+). | Candidate |

"Deprecated" = in the raw dataset but not in the model schema (PassengerId, Name, Ticket, Cabin) — decomposed into an engineered feature or dropped as a bare identifier.