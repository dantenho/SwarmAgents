// LangSwarm topology — idempotent (MERGE)
MERGE (a:Agent {name: "researcher"}) SET a.description = "You are a web research specialist.";
MERGE (a:Agent {name: "analyst"}) SET a.description = "You are a quantitative analyst.";
MERGE (a:Agent {name: "writer"}) SET a.description = "You are a technical writer.";
MATCH (s:Agent {name: "researcher"}), (t:Agent {name: "analyst"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
MATCH (s:Agent {name: "researcher"}), (t:Agent {name: "writer"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
MATCH (s:Agent {name: "analyst"}), (t:Agent {name: "researcher"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
MATCH (s:Agent {name: "analyst"}), (t:Agent {name: "writer"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
MATCH (s:Agent {name: "writer"}), (t:Agent {name: "researcher"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
MATCH (s:Agent {name: "writer"}), (t:Agent {name: "analyst"}) MERGE (s)-[:HANDS_OFF_TO]->(t);
