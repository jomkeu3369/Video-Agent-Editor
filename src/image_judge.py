import os
import base64
import mimetypes
import asyncio

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
import json

from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
load_dotenv()
logging.langsmith("image_judge")

llm = ChatOpenAI(model="gpt-5.6-luna", temperature=0.0,)


class JudgeResult(BaseModel):
    score: float = Field(description="0.0 ~ 1.0 사이의 점수")
    grade: Literal["good", "almost_good", "ambiguous", "almost_wrong", "wrong"] = Field(description="Saliency 결과 평가 등급")
    object_coverage: float = Field(description="주요 객체가 얼마나 잘 포함되었는지 0.0 ~ 1.0")
    noise_level: float = Field(description="불필요한 배경이나 노이즈가 포함된 정도 0.0 ~ 1.0")
    reason: str = Field(description="판단 이유")

judge_llm = llm.with_structured_output(JudgeResult)

class ImageJudgeState(TypedDict, total=False):
    path_1: str
    path_2: str

    score: float
    grade: str

    object_coverage: float
    noise_level: float

    reason: str


def image_to_data_url(path: str) -> str:

    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        mime_type = "image/png"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
    )

async def judge_image_node(state: ImageJudgeState) -> ImageJudgeState:
    path_1 = state["path_1"]
    path_2 = state["path_2"]

    image_1 = image_to_data_url(path_1)
    image_2 = image_to_data_url(path_2)

    system_prompt = """
You are an expert evaluator for Salient Object Detection.

You will receive two images.

Image 1:
The original image.

Image 2:
The result produced from a saliency detection pipeline.

Your task is NOT to judge whether the two images look visually identical.

Instead, determine whether Image 2 correctly captures the main semantically
important object from Image 1.

Evaluate:

1. Whether the main object in Image 1 is captured.
2. Whether important parts of the object are missing.
3. Whether excessive background or unrelated objects are included.
4. Whether Image 2 focuses on the wrong region.
5. Whether the detected region would be useful as an ROI for image
   classification or automatic video cropping.

Scoring:

1.0:
The main object is almost perfectly captured with little unnecessary background.

0.8:
The main object is correctly captured with minor missing areas or background noise.

0.6:
The correct object is mostly detected, but the result has noticeable problems.

0.4:
Only part of the correct object is detected or significant irrelevant regions are included.

0.0:
The wrong region/object is detected or the main object is almost completely missed.

Grade:

good:
score >= 0.9

almost_good:
0.7 <= score < 0.9

ambiguous:
0.5 <= score < 0.7

almost_wrong:
0.3 <= score < 0.5

wrong:
score < 0.3

object_coverage:
How completely the main object is preserved.

noise_level:
How much unnecessary background/noise is included.
0 means almost no noise.
1 means severe noise.
"""


    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """
Compare Image 1 and Image 2.

Image 1 is the original image.
Image 2 is the saliency detection result.

Evaluate whether the salient object from Image 1
was correctly detected in Image 2.
"""
            },

            {
                "type": "text",
                "text": "Image 1 - Original"
            },

            {
                "type": "image_url",
                "image_url": {
                    "url": image_1
                }
            },

            {
                "type": "text",
                "text": "Image 2 - Saliency Result"
            },

            {
                "type": "image_url",
                "image_url": {
                    "url": image_2
                }
            }
        ]
    )


    result = await judge_llm.ainvoke([SystemMessage(content=system_prompt), message])

    return {
        **state,

        "score": result.score,
        "grade": result.grade,

        "object_coverage": (
            result.object_coverage
        ),

        "noise_level": (
            result.noise_level
        ),

        "reason": result.reason
    }

builder = StateGraph(ImageJudgeState)
builder.add_node("judge_image", judge_image_node)
builder.add_edge(START, "judge_image")
builder.add_edge("judge_image", END)
graph = builder.compile()


async def main():
    success = 0
    fail = 0

    with open("src/data/judge_results.jsonl", "a", encoding="utf-8") as f:
        for idx in range(1, 101):
            result = await graph.ainvoke({
                "path_1": f"src/data/Original/cat_{idx}.png",
                "path_2": f"src/data/BiRefNet_HR_ROI/crops/crop_{idx}.png"
            })

            record = {
                "index": idx,
                "score": result["score"],
                "grade": result["grade"],
                "object_coverage": result["object_coverage"],
                "noise_level": result["noise_level"],
                "reason": result["reason"]
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            f.flush()

            if result["grade"] in ["good", "almost_good"]:
                success += 1

            elif result["grade"] in ["wrong", "almost_wrong"]:
                fail += 1

            print(
                f"[{idx}/100] "
                f"{result['grade']} "
                f"{result['score']:.2f}"
            )

            await asyncio.sleep(0.5)

    print(f"성공: {success}/100")
    print(f"실패: {fail}/100")

if __name__ == "__main__":
    asyncio.run(main())