CREATE (wtg_P04:WTG {name: 'P04'}),
       (model_E02:WTGModel {name: 'E92', oem: 'Enercon'}),
       (event_FatigueIncreased:Event {name: 'Fatigue increased', status: 'Pending', severity: 'HIGH_RISK_DEVELOPMENTS'}),
       (user_Yuri:User {name: 'Yuri Jean Fabris'}),
       (wtg_P04)-[:HAS_MODEL]->(model_E02),
       (wtg_P04)-[:HAS_EVENT]->(event_FatigueIncreased),
       (user_Yuri)-[:ASSIGNED_TO]->(event_FatigueIncreased)