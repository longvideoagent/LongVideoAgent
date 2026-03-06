# utils.py
from pathlib import Path
from PIL import Image
import io, base64, math, re
from openai import OpenAI
import os
import os, re, json, argparse, time
from typing import Dict, List, Any, Tuple
import random
import concurrent.futures

def convert_image_to_base64_data_url(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buf = io.BytesIO(); img.save(buf, format='JPEG')
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    """Build concatenated subtitles for one episode, sorted by clip key."""
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    sorted_clips = sorted(matching_clips.items())
    formatted_subtitles = []
    for clip_key, subtitle_text in sorted_clips:
        formatted_subtitles.append(f"<{clip_key}>{subtitle_text}</{clip_key}>")
    return "\n".join(formatted_subtitles)

def llm_generate(user_content: List[Dict], model: str = "grok-4-fast-reasoning") -> str:
    """
    Call the API-backed LLM and return response text.
    """
    grounding_api = os.getenv("qdd_api")
    grounding_client = OpenAI(
        api_key=grounding_api,
        base_url="https://api2.aigcbest.top/v1  "
    )
    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=2500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return "Error: Failed to generate response"

def get_episode_images(episode_prefix: str, base_frame_dir: str = "../Tvqa_data/bbt_frames") -> List[str]:
    """
    Select up to 16 evenly distributed middle-frame images for one episode.
    """
    episode_images = []
    clip_dirs = [d for d in os.listdir(base_frame_dir) if d.startswith(episode_prefix)]
    sorted_clip_dirs = sorted(clip_dirs)
    
    M = len(sorted_clip_dirs)
    if M == 0:
        print(f"Warning: No clips found for episode {episode_prefix}")
        return []
    
    num_images = min(16, M)
    if num_images < M:
        step = M / num_images
        selected_indices = [int(i * step) for i in range(num_images)]
    else:
        selected_indices = list(range(M))
    
    for idx in selected_indices:
        vid = sorted_clip_dirs[idx]
        middle_frame_num = 90
        img_path = Path(base_frame_dir, vid, f"{middle_frame_num:05d}.jpg")
        if img_path.is_file():
            url = convert_image_to_base64_data_url(str(img_path))
            if url:
                episode_images.append(url)
        else:
            print(f"Warning: Middle frame not found for clip {vid}")
    
    print(f"Selected {len(episode_images)} images for episode {episode_prefix}")
    return episode_images

def process_single_question(q: Dict, clip_subtitles: Dict[str, str], base_frame_dir: str = "../Tvqa_data/bbt_frames", model: str = "gemini-2.5-pro-exp-03-25") -> Dict[str, Any]:
    """
    Process one question with a simplified evaluation pipeline.
    """
    try:
        # Get the episode prefix from vid_name.
        episode_prefix = q["vid_name"].split('_')[0]  # Default format: bbt_xxx.
        
        # Build episode-level subtitles.
        episode_subtitles = build_subtitles_for_episode(clip_subtitles, episode_prefix)

        # Collect episode-level images.
        episode_image_urls = get_episode_images(episode_prefix, base_frame_dir)
        print(f"Loaded {len(episode_image_urls)} images for episode {episode_prefix}")
        
        # Build message content.
        messages_content = []
        
        # Add images.
        for url in episode_image_urls:
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        # Add text prompt.
        text = f"""
Question: {q['q']}
Options:
a0: {q.get('a0', '')}
a1: {q.get('a1', '')}
a2: {q.get('a2', '')}
a3: {q.get('a3', '')}
a4: {q.get('a4', '')}

Subtitles: {episode_subtitles}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Images are from the middle frame of each clip in the episode. Use the images and subtitles to answer the question.

Please provide the answer in the format: <answer>aX. Explanation</answer> where X is 0-4 and Explanation is your reasoning.

Example:
If the correct option is a4(sheldon) and the reason is because of some detail in the subtitles, output:
<answer>a4.sheldon</answer>
"""
        messages_content.append({
            "type": "text",
            "text": text
        })
        
        # Call LLM.
        response = llm_generate(messages_content, model)
        
        # Extract content inside <answer>...</answer>.
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        final_answer = answer_match.group(1).strip() if answer_match else "unknown"
        
        return {
            "vid": q["vid_name"],
            "question": q['q'],
            "final_answer": final_answer,
            "response": response
        }
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return {}

def process_question_wrapper(
    idx: int,
    q: Dict,
    total: int,
    clip_subtitles: Dict[str, str],
    base_frame_dir: str,
    model: str
) -> Dict[str, Any]:
    """
    Wrapper used by multithreaded execution.
    """
    print(f"Processing question {total}")
    record = process_single_question(q, clip_subtitles, base_frame_dir, model)
    if record:
        record["_question_idx"] = idx
    return record

def run_simple_qa(questions_path: str, subs_path: str, output_filename: str, 
                  base_frame_dir: str, model: str = "gemini-2.5-pro-exp-03-25", num_threads: int = 5) -> None:
    """
    Main flow for simplified API evaluation.
    """
    # Load inputs.
    clip_subtitles = json.load(open(subs_path, encoding='utf-8'))
    questions = json.load(open(questions_path, encoding='utf-8'))

    results: List[Dict[str, Any]] = []
    total = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    # Process questions concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit one future per question.
        futures = []
        for idx, q in enumerate(questions, start=1):
            total = idx
            future = executor.submit(
                process_question_wrapper, idx, q, total, clip_subtitles, base_frame_dir, model
            )
            futures.append(future)

        # Collect results.
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record:  # Success case.
                results.append(record)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}), stopping...")
                    break

    # Restore original question order for correct metric alignment.
    results = sorted(results, key=lambda x: x.get("_question_idx", 10**9))

    print(f"\nTotal processed: {total}")

    # Save simplified output.
    simplified_results = []
    for result in results:
        simplified_result = {
            "vid": result["vid"],
            "question": result["question"],
            "final_answer": result["final_answer"]
        }
        simplified_results.append(simplified_result)
    
    # Compute accuracy.
    correct_count = 0
    for result, q in zip(results, questions):
        gt_answer = f"a{q['answer_idx']}"  # Ground truth answer, e.g. "a0".
        # Extract leading aX token from final_answer.
        pred_match = re.match(r'(a[0-4])', result["final_answer"].strip().lower())
        pred_answer = pred_match.group(1) if pred_match else "unknown"
        if pred_answer == gt_answer.lower():
            correct_count += 1
    
    accuracy = correct_count / len(questions) if len(questions) > 0 else 0.0
    
    simplified_output = {
        "total": total,
        "metadata": {
            "accuracy": accuracy
        },
        "results": simplified_results
    }
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to {output_filename}")
    
    # Print summary stats.
    if simplified_results:
        metadata = simplified_output["metadata"]
        print(f"\n📊 Statistics:")
        print(f"Accuracy: {metadata['accuracy']:.2%}")
