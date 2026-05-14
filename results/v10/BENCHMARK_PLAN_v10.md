# BENCHMARK v10 FINAL PLAN
## Expert-Approved Final Design | 2026-05-13

---

## VERIFICATION RESULTS (from actual data analysis)

| Check | Result |
|-------|--------|
| JFK Flat Fare (mode) | **$70.00** (96,608 / 97.9% of RatecodeID=2 trips) |
| Cash tip = 0? | **100%** (TLC artifact: cash tips not recorded) |
| Rule 7 (cash_no_tip) | **VOID** -- replaced with `credit_no_tip` |
| LocationID range | **[1, 265]** -- no out-of-bounds |
| JFK Zone 217-229 pickup | 64,453 trips (actual JFK airport rides) |
| JFK Ratecode but Queens PU | Zone 132 = 71,855 trips (pre-scheduled airport shuttles) |

---

## 1. FEATURE VECTOR: 34D (SHARED)

All models use the SAME 34D vector. No feature engineering advantage.

| Index | Feature | Notes |
|-------|---------|-------|
| 0 | trip_distance | Raw |
| 1 | dur_min | Computed |
| 2 | fare_amount | Raw |
| 3 | passenger_count | Raw |
| 4 | total_amount | Raw |
| 5 | speed_mph | Computed |
| 6 | fare_per_mile | fare/distance |
| 7 | fare_per_min | fare/duration |
| 8 | fare_per_pax | fare/pax |
| 9 | hour | Raw datetime |
| 10 | day_of_week | Raw datetime |
| 11 | is_weekend | dow>=5 |
| **12** | **PU_Grid_X** | **NEW: pickup zone column** |
| **13** | **PU_Grid_Y** | **NEW: pickup zone row** |
| **14** | **DO_Grid_X** | **NEW: dropoff zone column** |
| **15** | **DO_Grid_Y** | **NEW: dropoff zone row** |
| 16 | fare_per_mile_norm | fare_per_mile / 2.5 |
| 17 | fare_per_min_norm | fare_per_min / 0.67 |
| 18 | speed_norm | speed / 12.0 |
| 19 | pax_per_mile | pax / distance |
| 20 | sin_hour | sin(2pi * hour / 24) |
| 21 | cos_hour | cos(2pi * hour / 24) |
| 22 | sin_dow | sin(2pi * dow / 7) |
| 23 | cos_dow | cos(2pi * dow / 7) |
| 24 | distance_squared | dist^2 (curvature) |
| 25 | RatecodeID=1 (Standard) | One-hot |
| 26 | RatecodeID=2 (JFK) | One-hot |
| 27 | RatecodeID=3 (Newark) | One-hot |
| 28 | RatecodeID=4 (Nassau) | One-hot |
| 29 | RatecodeID=5 (Negotiated) | One-hot |
| 30 | is_night | (>=20 OR <6) |
| 31 | log_fare | log(1 + fare) |
| 32 | log_distance | log(1 + distance) |
| 33 | inter_borough_rough | |PU_grid_Y - DO_grid_Y| |

**Why Grid X/Y instead of Borough:**
- Borough: 5 units (Manhattan, Bronx, Queens, Brooklyn, Staten) -- too coarse
- Grid: 16x17 = 272 cells -- micro-resolution (street/block level)
- Not redundant with RatecodeID (Ratecode = trip TYPE, Grid = trip LOCATION)
- **Critical for Type 3**: "$70 + RatecodeID=2 + PU_Grid(Queens, not JFK) = MISMATCH"

**Removed** (all redundant):
- is_rush_hour (duplicated by sin/cos hour)
- month (no value for short-term fraud)
- fare_distance_product (kNN synthesizes this)
- duration_per_distance (kNN synthesizes this)

---

## 2. 7 CANARY RULES (Layer 1 -- Static Filter)

| # | Rule Name | Logic | Target |
|---|-----------|-------|--------|
| 1 | negative_fare | fare_amount <= 0 | Gian lan thô |
| 2 | extreme_fare | fare_amount > 500 | Gian lan thô |
| 3 | high_fare_per_min | fare/duration_min > 5.0 | Gian lan thô |
| 4 | extreme_speed | speed_mph > 80 | Gian lan thô |
| 5 | phantom_trip | distance==0 AND fare>0 | Gian lan thô |
| 6 | invalid_passengers | pax < 1 OR pax > 6 | Gian lan thô |
| **7** | **credit_no_tip** | **payment_type==1 AND tip_amount==0** | **Borderline (NEW)** |

**Rule 7 fix**: Cash tip = 0 for 100% of cash payments (TLC artifact).
Valid signal: credit card (type=1) with zero tip = genuinely suspicious.

---

## 3. FRAUD TYPES (3 loai -- only inject into canary-clean records)

### Type 1: Short-Trip Meter Fraud (60%)
- Filter: `RatecodeID==1 AND distance < 1.0 mi AND canary_clean`
- Inject: `fare = $40-$80`
- Detection signal: "$80 for 0.5 mile is impossible for Standard ratecode"

### Type 2: Duration Manipulation (30%)
- Filter: `RatecodeID==1 AND distance 2-4 mi AND canary_clean`
- Inject: `duration *= 8-15x`
- Detection signal: "Time stretched 10x but money unchanged"

### Type 3: Ratecode Mismatch (10%)
- Filter: `RatecodeID==1 AND canary_clean`
- Inject: `RatecodeID=2, fare=EXACTLY $70.00`
- **CRITICAL**: Keep PULocationID as-is (Manhattan/downtown)
- Detection signal: `"$70 + JFK ratecode + Grid(not JFK zone) = MISMATCH"`
- Grid XY captures the spatial mismatch that makes this detectable

---

## 4. 10 NEIGHBORHOOD ADWIN (Layer 4 -- EIA)

| Neighborhood | LocationID range | Notes |
|-------------|-----------------|-------|
| manhattan | 1-43 | Manhattan core |
| brooklyn | 104-127 | Brooklyn |
| queens_lower | 128-148 | Queens lower |
| queens_upper | 149-161 | Queens upper |
| bronx | 44-103 | Bronx |
| staten_island | 162-181 | Staten Island |
| ewr | 182-196 | Newark Airport |
| jfk | 217-229 | JFK Airport |
| nalp | 230-234 | Nassau/Westchester |
| unknown | 235-265, 197-216 | Unknown zones |

---

## 5. 80 CONTEXT-BETA THRESHOLDS

10 neighborhoods x 8 cells = 80 unique thresholds.
Each threshold computed from warmup only. **Zero leakage.**

Cell = (Standard/Special) x (Day/Night) x (Weekday/Weekend)

---

## 6. HYBRID RENDEZVOUS ARCHITECTURE

```
[Input Stream] ---> [7 Canary Rules]
                          |
           +--------------+--------------+
           | (violation)  | (clean)       |
           v              v              v
    [Canary Branch]  [ML Branch: CA-MemStream-EIA]
    ANOMALY (0 ML)   34D -> kNN+AE -> 80 beta thresholds
           |              |
           +-------+------+
                   v
            [Voting: Canary overrides ML]
                   v
            [Final Decision]
```

**Scientific Integrity Claim:**
> "MemStream gốc và CA-MemStream-EIA dùng chung vector 34D. Không có chuyện 'ăn gian' thêm feature. Sự ưu việt của CA-MemStream-EIA nằm ở chỗ nó biết khi nào nên học (EIA/ADWIN-U) và khi nào nên dùng luật (Canary), giữ nguyên độ chính xác bắt lỗi tinh vi nhưng giảm 90% chi phí vận hành."

---

## 7. SCENARIOS

| Scenario | Canary | Type 1 | Type 2 | Type 3 | Total |
|----------|--------|--------|--------|--------|-------|
| canary_only | 5% | 0% | 0% | 0% | 5% |
| type1_only | 0% | 5% | 0% | 0% | 5% |
| type2_only | 0% | 0% | 5% | 0% | 5% |
| type3_only | 0% | 0% | 0% | 5% | 5% |
| mixed | 0% | 3% | 1.5% | 0.5% | 5% |
| **hybrid** | **2.5%** | **1.5%** | **0.75%** | **0.25%** | **5%** |

---

## 8. ALGORITHMS

| Algorithm | Description |
|-----------|-------------|
| Canary-Rules | 7 rules only, no ML |
| MemStream | Baseline ML streaming (100% budget) |
| CA-MemStream | Context-aware weighting + beta (100% budget) |
| CA-MemStream-EIA | CA-MemStream + 10 ADWIN + neighborhood beta (<5% budget) |
| Random | Baseline |

---

## 9. IMPLEMENTATION CHECKLIST

- [x] 34D shared feature vector (25D core + 4 Grid XY + 5 RatecodeID)
- [x] 7 Canary Rules (Rule 7 = credit_no_tip)
- [x] 3 Fraud Types with 60-30-10 weights
- [x] Type 3: JFK flat fare = $70.00 (verified)
- [x] 10 Neighborhood ADWIN instances
- [x] 80 Context-beta thresholds (10 neighborhoods x 8 cells)
- [x] `score_one()` uses context-beta for normalized scoring
- [x] `update_one()` uses neighborhood-aware beta
- [x] `inject_realistic_fraud()` with `.at[]` safe assignment
- [x] Hybrid scenario: 2.5% canary + 1.5% T1 + 0.75% T2 + 0.25% T3
