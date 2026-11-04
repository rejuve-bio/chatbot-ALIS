# A bounded review, uncertainty and correction demonstration

Use two authorised clinician accounts and one intervention, such as entry 7 in
`full_temp.json`. The sample judgments are for a showcase, not clinical advice.

1. The first account opens 7, selects Support and adds a demo note.
2. From the list choose **a** and sign in as the second account. Open 7; the first
   clinician's response is visible. Select Uncertain. The first Support still
   supplies strength 1 and count confidence 0.5; the uncertainty count and review
   status show that the assessment is unresolved.
3. On 7 choose **5**, select a concern category, and explain what needs checking.
   For example, “Check whether the supplied cancer entry is being confused with
   biomarker findings in one specific study population.” This records a concern
   under the second account; the paper and upstream entry are not rewritten.
4. If the second account changes Uncertain to Oppose, one Support and one Oppose
   produce strength 0.5 and review-count confidence 0.667. The old response remains
   in history and the clinician is counted once.
5. After addressing the issue, the second account clears their own concern using
   **5**, and can revise their assessment again through the ordinary review
   choices. Clearing the concern alone does not change their judgment.
6. Choose **v** on the list to inspect More support, Needs review or More
   opposition. Set a review-coverage minimum if desired. Nothing is deleted.
7. Restart and confirm that both identities, current judgments, concern history,
   updated ranking and source evidence remain available.

This demonstrates feedback changing the research shortlist and the next reviewer's
information. It does not require rewriting publication claims or interpreting
free-text reasons into invented numerical weights.
