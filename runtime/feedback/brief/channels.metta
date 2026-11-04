(= (initChannels) (change-state! &review-task-delivered False))
(= (receive) (if (get-state &review-task-delivered) "" (progn (change-state! &review-task-delivered True) "Explain the inspected evidence and reviewer discussion. Return a cited read-only review brief using finish-brief.")))
(= (send $msg) REVIEW-NOTIFICATION)
