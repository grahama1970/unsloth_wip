Grokking remains an active area of research in machine learning as of 2025, with recent studies expanding our understanding of its mechanisms and applications. While initially observed in shallow models, newer research demonstrates its relevance to deeper architectures and more complex tasks. Here's the current status:

### Grokking in 2025: Validity and Applications
1. **Continued Validation**  
   Recent papers ([1][3][5]) confirm grokking persists in modern architectures:
   - Transformer models exhibit grokking through distinct phases: memorization → circuit formation → alignment [1]
   - Deep MLPs (up to 12 layers) show enhanced grokking susceptibility compared to shallow networks [5]
   - New theoretical frameworks like ExPLAIND enable unified analysis of model components and training dynamics [1]

2. **Mechanistic Advances**  
   Current research identifies key factors:
   - Weight decay regularization remains crucial for triggering grokking [3]
   - Feature rank dynamics correlate with generalization phases in deep networks [5]
   - Transition from "lazy" to "rich" training regimes explains delayed generalization [6][7]

3. **Practical Implementations**  
   Emerging techniques leverage grokking principles:
   - Layer swapping accelerates generalization by reusing aligned components [1]
   - Initialization strategies from grokked models enable instant generalization [1]
   - Grokfast method achieves 50× speedup in algorithmic tasks [8]

### Limitations and Open Questions
- Primarily studied on synthetic/algorithmic tasks [2][7]
- Computational costs limit large-scale validation [1]
- No consensus on universal indicators for grokking onset [5]

The phenomenon you describe – extended training (30×+ epochs) with initial validation error increase followed by sudden generalization – remains recognized as **grokking**. Recent work shows this delayed generalization pattern persists in modern architectures when using appropriate regularization and initialization strategies [1][5][6]. While not yet mainstream in production systems, grokking continues to provide insights into neural network learning dynamics and generalization mechanisms.

Citations:
[1] https://arxiv.org/html/2505.20076v1
[2] https://www.quantamagazine.org/how-do-machines-grok-data-20240412/
[3] https://arxiv.org/pdf/2501.04697.pdf
[4] https://wandb.ai/byyoung3/mlnews3/reports/Grokking-Improved-generalization-through-over-overfitting--Vmlldzo4MjczMzgz
[5] https://arxiv.org/html/2405.19454v1
[6] https://arxiv.org/abs/2310.06110
[7] https://en.wikipedia.org/wiki/Grokking_(machine_learning)
[8] https://www.reddit.com/r/MachineLearning/comments/1defvmv/d_is_grokking_solved/
[9] https://arxiv.org/html/2502.01774v1
[10] https://www.lesswrong.com/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking
[11] https://openreview.net/forum?id=4rEI2JdHH6
[12] https://cacm.acm.org/news/training-neural-networks-to-grok/
[13] https://arxiv.org/abs/2505.20076
[14] https://spearhead.so/the-accidental-discovery-that-changed-ai-how-openai-grokked-llms/
[15] https://www.reddit.com/r/LLMDevs/comments/1ddvoi9/is_llm_grokking_the_new_hype_trend/
[16] https://galaxy.ai/youtube-summarizer/understanding-grokking-the-key-to-unlocking-llm-performance-SRfJQews1AU
[17] https://www.educative.io/answers/why-doesnt-validation-loss-decrease-after-certain-epochs
[18] https://stackoverflow.com/questions/65625426/validation-loss-is-keep-decreasing-while-training-loss-starts-to-increase-after

---
Answer from Perplexity: pplx.ai/share