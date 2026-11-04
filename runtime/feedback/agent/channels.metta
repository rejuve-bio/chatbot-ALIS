(= (initChannels) (change-state! &review-task-delivered False))
(= (receive) (if (get-state &review-task-delivered) "" (progn (change-state! &review-task-delivered True) "Interpret the reviewer reason using the inspected source packet. Propose a claim-specific assessment for clinician confirmation. Call finish-review.")))
(= (send $msg) REVIEW-NOTIFICATION)
