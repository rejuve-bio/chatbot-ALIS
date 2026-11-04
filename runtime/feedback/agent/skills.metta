!(import_prolog_functions_from_file (library Omega ./src/skills.pl) (first_char gc))
(= (getSkills) ("finish-review JSON"))
(= (read-review-context) (py-call (bridge.read_context)))
(= (finish-review $payload) (py-call (bridge.finish $payload)))
