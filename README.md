# Homa - Offline, On-device AI assistant for African Farmers
> Please find the full technical report in [REPORT.md](./REPORT.md)

## Running and profiling Homa
### Prerequisites
Please ensure you have the following installed:
1. [ADTC Profiler](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler)
2. [Ollama](https://ollama.com/download)

### How to run Homa Locally
1. Download Homa from Hugging Face
```bash
./download_model.sh
```

2. Build Homa locally
```bash
ollama create homa -f Modelfile
```

3. Run Homa
```bash
ollama run homa
```

### How to profile Homa for [ADTC 2026](https://adtc-2026.devpost.com/)
1. Run the [ADTC profiler](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler) from the root directory
```bash
adtc-profiler run \
  --submission . \
  --mode participant \
  --output submission.json
```

2. Output the profiler report in the terminal
```bash
cat submission.json
```
## Inspiration

The cost of access to frontier intelligence and intelligent tools has dropped steadily over the past few years and is projected to drop significantly in the coming years. This translates into greater productivity, strengthening the economy for those with the basic infrastructure needed to access these intelligent tools.

Research shows that the average African farmer lives in remote locations where access to the internet or high-tech devices is limited<sup>[1]</sup>. This suggests these farmers may not access frontier intelligence, no matter how cheap it becomes.

This fundamental barrier is what inspired the design direction we chose for Homa.

## What it does

Homa is an offline, on-device AI agricultural assistant equipped with practical, locally relevant guidance on an affordable laptop with no internet dependency. Homa can provide guidance to farmers on the following:

- Crop production
- Livestock production
- Pest and disease management
- Fertiliser decision
- Seasonal planting guidance for Nigerian/West-African context

## How we built it

We built Homa as an **offline, quantized AI assistant for Nigerian farmers**, optimized to run on modest hardware without requiring an internet connection.

We started with **AfriqueGemma-4B**, a model with African-language continued pretraining, and fine-tuned it using **LoRA/QLoRA** on our agricultural instruction dataset. We expanded the dataset to over **5,500 deduplicated examples across English, Yoruba, Hausa, and Igbo**, while continuously testing the model for agricultural accuracy, identity consistency, safety, and multilingual behavior.  

We then optimized the model for offline deployment using **GGUF quantization and llama.cpp**. We experimented with standard and importance-matrix-guided quantization to reduce memory usage while preserving accuracy.

The most important part of the development process was **real-hardware profiling**. AfriqueGemma-4B delivered strong multilingual capabilities, but its CPU throughput and thermal behavior created too much risk of failing the competition's hardware gate. We therefore evaluated smaller alternatives and ultimately moved to **Qwen2.5-1.5B-Instruct**, which already had instruction-following and native system-role support.  

For the final model, we fine-tuned Qwen using **LoRA**, trained only on the assistant's completion tokens, merged the adapter on CPU, and converted the resulting model to **F16 and Q4_K_M GGUF** for lightweight offline inference. The final candidate achieved **22+ tokens/sec, 72%+ benchmark accuracy, and under 2GB peak memory**.  

In summary, we built Homa through an iterative cycle of **data collection -> fine-tuning -> quantization -> real-hardware profiling -> evaluation -> optimization**, ultimately prioritizing a model that could reliably deliver useful agricultural intelligence within the hardware constraints of the target users.

## Challenges we ran into

The biggest challenge was **balancing model intelligence with the hardware constraints of the target environment**. Our initial AfriqueGemma-4B model had strong multilingual capabilities, but real-world profiling showed low and inconsistent CPU throughput, high first-token latency, and thermal-performance concerns. This created a serious risk of failing the automated hardware gate.

We also ran into **GPU memory and training infrastructure issues**. During training, our multi-GPU environment was unintentionally splitting the model across GPUs and slowing training significantly. We also encountered CUDA out-of-memory errors when merging the LoRA adapter, which we solved by explicitly freeing GPU memory and performing the merge on CPU.

Another major challenge was **data quality and model behavior**. The model sometimes incorrectly identified itself as being created by OpenAI, used repetitive refusal patterns, and behaved differently across languages. For example, it could provide treatment advice in English but refuse an equivalent question in Hausa. We had to repeatedly audit, deduplicate, and regenerate training examples to address these issues.  
We also discovered that **RAG did not automatically improve performance**. In some scored cases, retrieval returned irrelevant information, which actually caused the model to produce worse answers than it did without RAG. This forced us to treat retrieval quality as a separate engineering problem rather than assuming that adding RAG would improve the system.

Ultimately, our biggest challenge was making the right **engineering trade-off**. We had to choose between a larger multilingual model with a potential scoring advantage and a smaller model that could reliably run within the hardware constraints. We ultimately chose Qwen2.5-1.5B-Instruct because passing the hardware gate was more important than retaining the African-language multiplier.

## Accomplishments that we're proud of

We’re proud that we took Homa from an initial model idea to a **fully quantized, offline agricultural assistant** through multiple rounds of training, evaluation, and optimization. We built and tested our own multilingual agricultural dataset, fine-tuned multiple models, and produced deployable GGUF versions for low-resource hardware.

We’re particularly proud of the **engineering rigor behind the final model**. Rather than optimizing only for benchmark numbers, we profiled the models on real hardware, identified throughput and memory bottlenecks, experimented with different quantization strategies, and ultimately delivered a model achieving **22+ tokens/sec, 72%+ benchmark accuracy, and under 2GB peak memory**.

Most importantly, we’re proud of the willingness to **change direction when the evidence demanded it**. We moved away from a more capable multilingual 4B model to a smaller 1.5B model because we determined that reliably running on the target hardware mattered more than theoretical capability.

## What we learned

The biggest lesson was that **building an AI system for real-world constraints is very different from simply fine-tuning a model**. A model can perform well during training but still fail when deployed because of latency, memory, thermal behaviour, or hardware limitations. Real-hardware profiling therefore needs to be part of the development loop, not something done at the end.

We also learned that **better data can matter more than simply adding more data**. We encountered identity inconsistencies, repetitive refusal pattern, cross-lingual behavior differences, and duplicate examples. Systematically auditing and improving the dataset was essential to getting more reliable behavior.

Another important lesson was that **adding complexity does not necessarily improve a system**. Our RAG pipeline actually made some scored responses worse because retrieval returned irrelevant context. This taught us to validate every component independently instead of assuming that techniques like RAG will automatically improve model performance.

Finally, we learned that **engineering is fundamentally about trade-offs**. The best model on paper is not necessarily the best model for the problem. For Homa, reliability, efficiency, and deployability ultimately mattered more than maximising multilingual capability or model size.

## What's Next for Homa

Our next step is to move Homa from a strong offline assistant into a more complete **AI platform for farmers**.

First, we plan to **complete the RAG pipeline**, improving retrieval quality so Homa can ground its answers in reliable, up-to-date agricultural information rather than relying entirely on what is stored in the model's parameters.

We also want to add **multimodal capabilities**, allowing farmers to provide information beyond text. For example, Homa could combine agricultural knowledge with **current market conditions to recommend appropriate prices for their crops**, helping farmers make better decisions about when and where to sell.

On the deployment side, we plan to **bundle Homa into a native desktop application**, making the entire system easy to install and use offline without requiring users to manage models, dependencies, or command-line tools themselves.

Finally, we want to develop a **much smaller mobile version of Homa** that can run directly on farmers' phones. This is particularly important for our target users because the long-term goal is to make useful AI accessible without requiring expensive hardware or constant internet connectivity.

## Reference
1 - [TechAfrica News (2026), _2025 vs 2026 Mobile Industry Checkpoint: Has Anything Actually Changed for Africa?_, March 5, 2026.](https://techafricanews.com/2026/03/05/2025-vs-2026-mobile-industry-checkpoint-has-anything-actually-changed-for-africa/)
