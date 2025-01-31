!pip install gradio
!pip install groq

import gradio as gr
import groq
import os
os.environ['GROQ_API_KEY'] = ##

def get_score(ground_truth, generated_response):
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
            messages=[
            {
                "role": "system",
                "content": "you are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Rate the response based on the ground truth. Give scores on truthfulness, groundedness, and hallucinations out of 5. The more the better. Ground truth: {ground_truth} Response: {generated_response}. Format output as Truthfulness:{{}} Groundedness:{{}} Hallucination:{{}}",
            }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content

demo = gr.Interface(
    fn=get_score,
    inputs=["text", "text"],
    outputs=["text"],
)
demo.launch()