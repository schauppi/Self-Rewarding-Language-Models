prompt_step_01 = """

Come up with a series of tasks and questions. Only the task/question,
no further text/explanation, no additional information.
The task or question should be something a person would ask a chatbot.

"""

judge_prompt = """Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. 

The user's question is between <question> and </question>
The response of the AI Assistant is between <response> and </response>

Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
- If the response repeats itself or is not concise and to the point, score the response 0.

<question>{prompt}</question>
<response>{response}</response>

After examining the user’s instruction and the response:
- output the score of the evaluation using this exact format: "score: <total points>", where <total points> is between 0 and 5
- Briefly justify your total score, up to 100 words.
"""
