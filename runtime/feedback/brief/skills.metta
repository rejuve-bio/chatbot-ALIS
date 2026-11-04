!(import_prolog_functions_from_file (library Omega ./src/skills.pl) (first_char gc))
(= (getSkills) ("finish-brief JSON"))
(= (read-brief-context) (py-call (bridge.read_context)))
(= (finish-brief $payload) (py-call (bridge.finish $payload)))
