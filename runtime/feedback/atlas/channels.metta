(= (initChannels) (change-state! &atlas-task-delivered False))
(= (receive) (if (get-state &atlas-task-delivered) "" (progn (change-state! &atlas-task-delivered True) "Assemble the current reviewed intervention atlas.")))
(= (send $msg) ATLAS-NOTIFICATION)
