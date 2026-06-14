"""
ACA-DLM v6c: Large prompt set - 256 QA-style prompts.

Uses diverse factual/creative questions similar to our original 32,
but expanded to 256 for statistical robustness.
Batch resumable: processes 64 prompts per invocation.
"""
import os, sys, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm
from ttt_dllm_v3_sweep import load_model
from aca_dllm_v2_sweep import remask_retry_v2

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 256
SEED = 42


def make_256_prompts(tokenizer):
    """Generate 256 diverse QA-style prompts."""
    questions = [
        # Science (40)
        "What is the theory of general relativity and how does it differ from special relativity?",
        "Explain how photosynthesis works in plants and why it is essential for life on Earth.",
        "What are the main differences between DNA and RNA in terms of structure and function?",
        "How do black holes form and what happens at the event horizon?",
        "Describe the process of nuclear fusion and its potential as an energy source.",
        "What is quantum entanglement and why did Einstein call it spooky action at a distance?",
        "How does the human immune system recognize and fight pathogens?",
        "What are stem cells and why are they important for medical research?",
        "Explain the concept of entropy and the second law of thermodynamics.",
        "What causes earthquakes and how are they measured on the Richter scale?",
        "How do neurons communicate with each other in the brain?",
        "What is CRISPR gene editing and how might it change medicine?",
        "Describe the structure of an atom and the forces that hold it together.",
        "What is dark matter and why do scientists believe it exists?",
        "How does evolution by natural selection work according to Darwin?",
        "What are the different states of matter and what determines transitions between them?",
        "Explain how vaccines work to protect against infectious diseases.",
        "What is the Heisenberg uncertainty principle in quantum mechanics?",
        "How do telescopes work and what are the different types?",
        "What is the periodic table and how are elements organized in it?",
        "Describe the carbon cycle and its importance for climate regulation.",
        "What are the fundamental forces of nature and how do they differ?",
        "How does the greenhouse effect work and what gases contribute to it?",
        "What is plate tectonics and how does it shape the Earth's surface?",
        "Explain how antibiotics work and why antibiotic resistance is a concern.",
        "What is the Doppler effect and how is it used in astronomy?",
        "How do magnets work at the atomic level?",
        "What is the difference between speed and velocity in physics?",
        "Describe how the water cycle works and its role in weather patterns.",
        "What are prions and why are they unusual compared to other pathogens?",
        "How does radiocarbon dating work to determine the age of artifacts?",
        "What is the Standard Model of particle physics?",
        "Explain how solar panels convert sunlight into electricity.",
        "What are the main branches of mathematics and how do they relate?",
        "How does the human digestive system break down food into nutrients?",
        "What is the Cambrian explosion and why is it significant in evolution?",
        "Describe the life cycle of a star from formation to death.",
        "What is superconductivity and what are its potential applications?",
        "How do batteries store and release electrical energy?",
        "What is the microbiome and how does it affect human health?",
        # History (40)
        "What were the main causes of World War I and how did it reshape Europe?",
        "Describe the rise and fall of the Roman Empire and its lasting legacy.",
        "What was the Industrial Revolution and how did it transform society?",
        "How did the printing press invented by Gutenberg change the world?",
        "What were the key events of the French Revolution and their significance?",
        "Describe the ancient Egyptian civilization and their major achievements.",
        "What was the Cold War and how did it influence global politics?",
        "How did the Renaissance transform art, science, and culture in Europe?",
        "What were the causes and consequences of the American Civil War?",
        "Describe the Silk Road and its importance in connecting civilizations.",
        "What was the Age of Exploration and which countries led it?",
        "How did ancient Greek democracy work and how does it differ from modern democracy?",
        "What were the Crusades and what impact did they have on Europe and the Middle East?",
        "Describe the Mongol Empire under Genghis Khan and its extent.",
        "What was the Enlightenment and how did it influence modern political thought?",
        "How did the abolition of slavery occur in different countries?",
        "What were the major dynasties of China and their contributions?",
        "Describe the Viking Age and their exploration of the North Atlantic.",
        "What was the Byzantine Empire and how did it preserve Roman culture?",
        "How did the Black Death affect European society in the 14th century?",
        "What was the significance of the Magna Carta in English history?",
        "Describe the Ottoman Empire at its peak and its eventual decline.",
        "What were the main events of World War II in the Pacific theater?",
        "How did the space race between the US and USSR unfold?",
        "What was the Meiji Restoration and how did it modernize Japan?",
        "Describe the ancient Mayan civilization and their calendar system.",
        "What were the causes of the Great Depression of the 1930s?",
        "How did the Berlin Wall come to be built and later fall?",
        "What was the Scientific Revolution and who were its key figures?",
        "Describe the history of the Olympic Games from ancient Greece to today.",
        "What was the Reformation and how did it change Christianity?",
        "How did the British Empire expand and eventually dissolve?",
        "What were the main achievements of the Islamic Golden Age?",
        "Describe the history of democracy in India since independence.",
        "What was the significance of the Treaty of Westphalia?",
        "How did writing systems develop in different civilizations?",
        "What were the Opium Wars and their impact on China?",
        "Describe the history of human rights movements in the 20th century.",
        "What was the Marshall Plan and how did it help rebuild Europe?",
        "How did the Internet develop from a military project to a global network?",
        # Technology (40)
        "How do large language models like GPT work at a high level?",
        "What is blockchain technology and how does it enable cryptocurrencies?",
        "Explain how the Internet works from typing a URL to loading a webpage.",
        "What is machine learning and how does it differ from traditional programming?",
        "How do self-driving cars use sensors and algorithms to navigate?",
        "What is cloud computing and what are its main service models?",
        "Describe how encryption works to protect data security.",
        "What is quantum computing and how does it differ from classical computing?",
        "How do search engines like Google rank web pages?",
        "What are neural networks and how do they learn from data?",
        "Explain how GPS technology determines your location on Earth.",
        "What is the Internet of Things and how is it changing everyday life?",
        "How do touchscreens work and what are the different types?",
        "What is 5G technology and how is it different from 4G?",
        "Describe how computer memory works from registers to hard drives.",
        "What is augmented reality and how does it differ from virtual reality?",
        "How do recommendation systems like those used by Netflix work?",
        "What is edge computing and why is it important for IoT?",
        "Explain how digital cameras capture and process images.",
        "What is natural language processing and what are its applications?",
        "How do operating systems manage computer resources?",
        "What is cybersecurity and what are the main types of threats?",
        "Describe how wireless charging technology works.",
        "What is DevOps and how does it improve software development?",
        "How do databases store and retrieve information efficiently?",
        "What is computer vision and how is it used in practice?",
        "Explain how fiber optic cables transmit data using light.",
        "What is robotics and what are the current limitations of robots?",
        "How does speech recognition technology work?",
        "What is the difference between HTTP and HTTPS protocols?",
        "Describe how modern CPUs are designed and manufactured.",
        "What is containerization in software and how does Docker work?",
        "How do electric vehicles work and what are their advantages?",
        "What is reinforcement learning and how is it used in game playing?",
        "Explain how social media algorithms decide what content to show.",
        "What is biometric authentication and what methods are used?",
        "How do drones fly and what technologies enable autonomous flight?",
        "What is serverless computing and when should it be used?",
        "Describe how video compression algorithms reduce file sizes.",
        "What is transfer learning in deep learning and why is it useful?",
        # Philosophy & Culture (30)
        "What is the trolley problem and what does it reveal about moral reasoning?",
        "Explain Plato's allegory of the cave and its modern relevance.",
        "What is existentialism and who were its main proponents?",
        "How does Confucianism influence East Asian societies today?",
        "What is the philosophy of mind and the hard problem of consciousness?",
        "Describe the main ideas of utilitarianism and its criticisms.",
        "What is stoicism and how can its principles be applied today?",
        "How did Buddhism spread from India across Asia?",
        "What is the social contract theory and who developed it?",
        "Explain the concept of the sublime in aesthetic philosophy.",
        "What is epistemology and what are the main theories of knowledge?",
        "How does music affect the brain and emotions?",
        "What is the difference between deductive and inductive reasoning?",
        "Describe the main schools of ancient Greek philosophy.",
        "What is phenomenology and how does it approach consciousness?",
        "How has the concept of human rights evolved over centuries?",
        "What is the relationship between language and thought?",
        "Explain the concept of the social construction of reality.",
        "What is game theory and how is it applied in economics?",
        "How do different cultures approach the concept of time?",
        "What is the philosophy of science and the demarcation problem?",
        "Describe the influence of mythology on modern storytelling.",
        "What is cognitive dissonance and how do people resolve it?",
        "How does architecture reflect the values and beliefs of a society?",
        "What is the difference between ethics and morality?",
        "Explain the concept of the Overton window in political discourse.",
        "What is semiotics and how do signs create meaning?",
        "How do different philosophical traditions view the nature of reality?",
        "What is the prisoner's dilemma and what does it teach about cooperation?",
        "Describe the main ideas of pragmatism as a philosophical movement.",
        # Nature & Geography (30)
        "What are the world's major biomes and how do they differ?",
        "How do coral reefs form and why are they important ecosystems?",
        "What causes the seasons on Earth and why do they differ by hemisphere?",
        "Describe the Amazon rainforest and its role in global ecology.",
        "What are tectonic plates and how do they cause geological features?",
        "How do rivers shape landscapes over geological time?",
        "What is biodiversity and why is it important for ecosystem stability?",
        "Describe the ocean currents and their influence on climate.",
        "What are the different types of clouds and what weather do they predict?",
        "How do animals migrate and what triggers their seasonal movements?",
        "What is desertification and what are its main causes?",
        "Describe the Arctic and Antarctic regions and how they differ.",
        "What are volcanoes and what happens during an eruption?",
        "How do ecosystems recover after natural disasters?",
        "What is the water table and why is groundwater important?",
        "Describe the formation and types of mountains around the world.",
        "What are invasive species and how do they threaten native ecosystems?",
        "How does the moon influence tides on Earth?",
        "What is soil and what determines its fertility?",
        "Describe the Great Barrier Reef and the threats it faces.",
        "What causes auroras and where can they be seen?",
        "How do caves form and what geological features are found inside?",
        "What is permafrost and what happens when it melts?",
        "Describe the Sahara Desert and how it has changed over millennia.",
        "What are wetlands and what ecological services do they provide?",
        "How do wind patterns form and influence global weather?",
        "What is the Ring of Fire and why is it geologically active?",
        "Describe the major types of forests and their characteristics.",
        "What is ocean acidification and how does it affect marine life?",
        "How do glaciers form, move, and shape the landscape?",
        # Creative & Practical (76 to reach 256)
        "Write a short story about a scientist who discovers a new element.",
        "Explain how to make bread from scratch, including the science behind it.",
        "What makes a good leader and how can leadership skills be developed?",
        "Describe a day in the life of an astronaut on the International Space Station.",
        "What is the history of chocolate from ancient Mesoamerica to today?",
        "How does the stock market work and what determines stock prices?",
        "Write about a future city powered entirely by renewable energy.",
        "What is the psychology of habit formation and how can bad habits be broken?",
        "Describe the process of making a feature film from script to screen.",
        "What are the health benefits and risks of different types of exercise?",
        "How does the human brain process visual information?",
        "What is the history of the English language and how has it evolved?",
        "Describe the science behind cooking and why different methods matter.",
        "What is emotional intelligence and how does it affect relationships?",
        "How do different economic systems compare in theory and practice?",
        "Write about a world where humans can communicate with animals.",
        "What is the science of sleep and why is it essential for health?",
        "Describe the invention of the telephone and its impact on communication.",
        "How does the human body maintain its temperature in different climates?",
        "What are the principles of good design in everyday objects?",
        "Explain how maps are made and why different projections distort reality.",
        "What is the history of medicine from ancient times to modern healthcare?",
        "How do different cultures celebrate coming-of-age traditions?",
        "What is the science behind fireworks and their colorful displays?",
        "Describe the process of wine making from grape to glass.",
        "How does memory work in the human brain and why do we forget?",
        "What is the history of mathematics from ancient Babylon to today?",
        "Explain how airplanes achieve flight and stay airborne.",
        "What are the different types of renewable energy and their tradeoffs?",
        "Describe the ecology of a tide pool and the organisms found there.",
        "How does the global food supply chain work from farm to table?",
        "What is the placebo effect and what does it tell us about the mind?",
        "Describe the development of writing from pictographs to alphabets.",
        "What is the science of color and how do humans perceive different wavelengths?",
        "How do bridges work and what are the main types of bridge designs?",
        "What is the history of space exploration and its major milestones?",
        "Explain the principles of photography from film to digital.",
        "How does the human ear process sound and distinguish different frequencies?",
        "What are the causes and effects of urbanization around the world?",
        "Describe the process of recycling different materials like plastic and glass.",
        "What is behavioral economics and how does it challenge classical theory?",
        "How do different musical instruments produce sound?",
        "What is the history of democracy and its various forms around the world?",
        "Explain how radar technology works and its applications.",
        "What is the science of nutrition and what constitutes a balanced diet?",
        "Describe the phenomenon of bioluminescence in nature.",
        "How does international trade work and what role do tariffs play?",
        "What is the history of animation from early cartoons to modern CGI?",
        "Explain how earthquakes are predicted and what warning systems exist.",
        "What are the main theories about the origin of life on Earth?",
        "How do different types of clocks measure time?",
        "What is the science behind optical illusions and why do they fool us?",
        "Describe the history of the automobile from the first cars to electric vehicles.",
        "How does the human body heal wounds and repair damaged tissue?",
        "What is the history of coffee and how did it become a global beverage?",
        "Explain how submarines work and can travel underwater.",
        "What are the principles of sustainable agriculture?",
        "Describe the science of weather forecasting and its accuracy.",
        "How do languages differ in their grammar and what are universals?",
        "What is the science of acoustics and how are concert halls designed?",
        "Describe the process of making paper from trees.",
        "How do different voting systems work and what are their strengths?",
        "What is the history of the Olympic Games and how have they changed?",
        "Explain how elevators work and the safety mechanisms they use.",
        "What are the psychological effects of social media on mental health?",
        "Describe the formation of fossils and what they tell us about the past.",
        "How does the postal system deliver mail across countries?",
        "What is the science behind perfume and how scents are created?",
        "Describe the major philosophical approaches to the meaning of life.",
        "How do computers generate random numbers and are they truly random?",
        "What is the history of tea and its role in different cultures?",
        "Explain how skyscrapers are designed to withstand wind and earthquakes.",
        "What is the microplastics problem and how does it affect the environment?",
        "Describe how the human eye works and common vision problems.",
        "How do different preservation methods keep food safe for longer?",
        "What is the history and science of navigation at sea?",
        "Explain how 3D printing works and its potential applications.",
        "What is the psychology of creativity and can it be taught?",
    ]

    assert len(questions) >= N_PROMPTS, f"Only {len(questions)} questions, need {N_PROMPTS}"

    prompts = []
    for q in questions[:N_PROMPTS]:
        # Apply chat template like the original 32 prompts
        chat = [{"role": "user", "content": q}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        prompts.append(tokens)

    return prompts


def load_progress(method_name):
    fname = RESULTS_DIR / f"v6c_{method_name}_progress.json"
    if fname.exists():
        with open(fname) as f:
            return json.load(f)
    return {"texts": [], "confs": [], "n_done": 0}


def save_progress(method_name, progress):
    fname = RESULTS_DIR / f"v6c_{method_name}_progress.json"
    with open(fname, "w") as f:
        json.dump(progress, f)


def run_generation_batch(method_name, batch_size=64):
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_256_prompts(tokenizer)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    progress = load_progress(method_name)
    start_idx = progress["n_done"]

    if start_idx >= N_PROMPTS:
        print(f"{method_name}: All {N_PROMPTS} prompts done")
        return progress

    end_idx = min(start_idx + batch_size, N_PROMPTS)
    print(f"\n{method_name}: Processing prompts {start_idx}-{end_idx-1} of {N_PROMPTS}")

    torch.manual_seed(SEED + start_idx)
    torch.cuda.manual_seed(SEED + start_idx)

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if method_name == "vanilla":
            sampler = dllm.core.samplers.MDLMSampler(
                model=model, tokenizer=tokenizer,
                scheduler=dllm.core.schedulers.LinearAlphaScheduler())
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
        elif method_name == "retry_70pct":
            seq, _ = remask_retry_v2(model, tokenizer, prompt, config,
                                     n_retries=2, remask_ratio=0.7, refine_steps=32)

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        progress["texts"].append(text)

        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        progress["confs"].append(gen_conf)

        if (pi - start_idx + 1) % 16 == 0:
            print(f"  {pi+1}/{N_PROMPTS} done")

    progress["n_done"] = end_idx
    save_progress(method_name, progress)
    print(f"  Batch done. Total: {end_idx}/{N_PROMPTS}")
    return progress


def run_ppl_eval(method_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress = load_progress(method_name)
    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS}, need more generation")
        return None

    ppl_file = RESULTS_DIR / f"v6c_{method_name}_ppls.json"
    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    print(f"\nPPL eval for {method_name} ({len(progress['texts'])} texts)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    eval_model.eval()

    ppls = []
    for ti, text in enumerate(progress["texts"]):
        if len(text) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))
        if (ti + 1) % 64 == 0:
            valid = [p for p in ppls if p is not None]
            print(f"  {ti+1}/{len(progress['texts'])}, mean PPL={np.mean(valid):.3f}")

    del eval_model
    torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "method": method_name, "n_prompts": N_PROMPTS, "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)), "ppl_std": float(np.std(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)), "conf_mean": float(np.mean(progress["confs"])),
        "all_ppls": valid_ppls,
    }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {method_name}: PPL={result['ppl_mean']:.3f} ± {result['ppl_std']:.3f}")
    return result


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "auto"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    if method == "auto":
        for m in ["vanilla", "retry_70pct"]:
            p = load_progress(m)
            if p["n_done"] < N_PROMPTS:
                method = m
                break
        else:
            for m in ["vanilla", "retry_70pct"]:
                if not (RESULTS_DIR / f"v6c_{m}_ppls.json").exists():
                    method = f"eval_{m}"
                    break
            else:
                method = "summary"

    if method == "summary":
        from scipy import stats as scipy_stats
        v = json.load(open(RESULTS_DIR / "v6c_vanilla_ppls.json"))
        r = json.load(open(RESULTS_DIR / "v6c_retry_70pct_ppls.json"))
        delta = (r["ppl_mean"] - v["ppl_mean"]) / v["ppl_mean"] * 100
        n = min(len(v["all_ppls"]), len(r["all_ppls"]))
        _, p = scipy_stats.ttest_rel(v["all_ppls"][:n], r["all_ppls"][:n])
        print(f"\n{'='*60}")
        print(f"256-PROMPT RESULTS")
        print(f"  Vanilla:    PPL={v['ppl_mean']:.3f} ± {v['ppl_std']:.3f}")
        print(f"  Retry 70%:  PPL={r['ppl_mean']:.3f} ± {r['ppl_std']:.3f}")
        print(f"  Delta:      {delta:+.1f}%")
        print(f"  p-value:    {p:.10f}")
        print(f"  N:          {n}")
        print(f"{'='*60}")
    elif method.startswith("eval_"):
        run_ppl_eval(method[5:])
    else:
        run_generation_batch(method, batch_size)
        p = load_progress(method)
        if p["n_done"] >= N_PROMPTS:
            print(f"\n{method} generation complete! Running PPL eval...")
            run_ppl_eval(method)


if __name__ == "__main__":
    main()
