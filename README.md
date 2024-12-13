HYDRA_FULL_ERROR=1 python intent_slot_classification.py --config-path conf/config --config-name intent_slot_classification_config.yaml

- https://turing.com/kb/natural-language-processing-function-in-ai
- https://spacy.io/usage/processing-pipelines
- https://www.analyticsvidhya.com/blog/2022/06/an-end-to-end-guide-on-nlp-pipeline/
- https://www.geeksforgeeks.org/natural-language-processing-nlp-pipeline/?ref=asr6
- https://www.freecodecamp.org/news/tag/nlp/
- https://www.deeplearning.ai/resources/natural-language-processing/
- https://cloud.google.com/natural-language
- https://www.ibm.com/topics
- https://roadmap.sh/ai-engineer
- https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html
- https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=
- https://developers.google.com/s/results/machine-learning/?q=Natural%20Language
- https://aws.amazon.com/ai/generative-ai/
- https://ai.meta.com/blog/when-to-fine-tune-llms-vs-other-techniques/
- https://colab.research.google.com/github/NVIDIA/NeMo/blob/r1.0.0rc1/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb#scrollTo=knF6QeQQdMrH
- https://colab.research.google.com/github/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/ml_ops/stage6/get_started_with_nvidia_triton_serving.ipynb#scrollTo=7326e903d212
- https://learn.microsoft.com/en-us/training/browse/?resource_type=learning%20path&expanded=data-ai&subjects=artificial-intelligence
- https://www.kdnuggets.com/tag/natural-language-processing
- https://colab.research.google.com/github/cleanlab/cleanlab-tools/blob/master/few_shot_prompt_selection/few_shot_prompt_selection.ipynb#scrollTo=e7da26fe
- https://colab.research.google.com/github/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb#scrollTo=FuXIFTFapAMI
- https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.ipynb#scrollTo=IjcflifmeKTE
- https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=GLFivpkwW1HY
- https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb

- LLMOps MLOps
      - LLMs as Operating Systems: Agent Memory
      - Serverless LLM Apps Amazon Bedrock
      - Efficiently Serving LLMs
- Transformers
      - Reinforcement Learning From Human Feedback
- Prompt Engineering
     - ChatGPT Prompt Engineering for Developers
     - Building Systems with the ChatGPT API
     - Pair Programming with a Large Language Model
     - Prompt Engineering with Llama 2&3
- Gen AI Applications Models
     - Getting Started with Mistral
     - Multi AI Agent Systems with crewAI
     - Build LLM Apps with LangChain.js
     - Red Teaming LLM Applications
- Task Automation
     - Pretraining LLMs
     - Retrieval Optimization: Tokenization to Vector Quantization
- Agents
      - Functions, Tools and Agents with LangChain
      - Serverless Agentic Workflows with Amazon Bedrock
  
- RAG
      - How Business Thinkers Can Start Building AI Plugins With Semantic Kernel
      - Building Agentic RAG with Llamaindex
      - JavaScript RAG Web Apps with LlamaIndex
      - AI Agentic Design Patterns with AutoGen
      - Building AI Applications With Haystack
      - Building and Evaluating Advanced RAG
      - Knowledge Graphs for RAG
- Computer Vision
      - Preprocessing Unstructured Data for LLM Applications
- Vector Databases
       - Vector Databases: from Embeddings to Applications
       - Building Applications with Vector Databases
- Embeddings
       - LangChain Chat with Your Data
       - Understanding and Applying Text Embeddings
       - Function-calling and data extraction with LLMs
- Evaluation and Monitoring
       - Evaluating and Debugging Generative AI
       - Automated Testing for LLMOps
- Fine Tuning
       - Finetuning Large Language Models
       - Improving Accuracy of LLM Applications
- Multimodal
       - Building Multimodal Search and RAG
       - Quantization Fundamentals with Hugging Face
       - Large Multimodal Model Prompting with Gemini
- Search and Retrieval
      - Advanced Retrieval for AI with Chroma
      - Large Language Models with Semantic Search
      - JavaScript RAG Web Apps with LlamaIndex
- AI Frameworks
- NLP
     - Understanding and Applying Text Embeddings
     - Large Language Models with Semantic Search(Cohere)
     - Safe and reliable AI via guardrails(Guardrails AI)
     - Open Source Models with Hugging Face
     - Multimodal RAG: Chat with Videos
  
- Diffusion Models
    - How Diffusion Models Work
    - 
- Document Processing
    - Preprocessing Unstructured Data for LLM Applications
    - AI Agents in LangGraph

- Event Driven AI
- On Device AI
- Time Series
- Deep and Machine Learning
- Data Engineering
- Convolution Neural Networks
- NLP with classification and vector spaces
- Deeplearning AI Tensorflow Developer

  What is an AI Agent?
An AI agent is a software or hardware entity that performs actions autonomously, with the goal of achieving specific objectives. It operates by perceiving its environment, processing information, making decisions, and taking actions based on its perceptions and goals.

Types of AI Agents
Simple Reflex Agents
Model-Based Reflex Agents
Goal-Based Agents
Utility-Based Agents
Learning Agents
Multi-Agent Systems

Problem Solving in AI
1. Search Algorithms in AI
Searching algorithms in artificial intelligence play a fundamental role by providing systematic methods for navigating through vast solution spaces to find optimal or satisfactory solutions to problems. These algorithms operate on various data structures, such as graphs or trees, to explore possible paths and discover solutions efficiently.

Searching algorithms are integral components in problem-solving, pathfinding, and optimization tasks across diverse AI applications, enabling systems to make decisions and find effective solutions in complex and dynamic environments. The choice of a specific searching algorithm depends on the characteristics of the problem domain, the available information, and the desired balance between computational efficiency and solution optimality.

1. Uninformed Search Algorithm
Uninformed search algorithms, also known as blind search algorithms, explore the search space without any domain-specific knowledge beyond the problem's definition. These algorithms do not use any additional information like heuristics to guide the search. Below is a list of common uninformed search algorithms:

Breadth-First Search (BFS)
Depth-First Search (DFS)
Uniform Cost Search (UCS)
Iterative Deepening Search
Bidirectional search
2. Informed Search Algorithm
Informed search algorithms, also known as heuristic search algorithms, use additional information (heuristics) to make decisions about which paths to explore. This helps in efficiently finding solutions by guiding the search process towards more promising paths. Here’s a list of common informed search algorithms:

Greedy Best-First Search
A Search* Algorithm
Recursive Best-First Search (RBFS)
Simplified Memory-Bounded A* (SMA*)
2. Local Search Algorithms
Local search algorithms in AI are optimization techniques that operate on a single current state (or a small set of states) and attempt to improve it incrementally by exploring neighboring states. They are particularly useful for solving optimization problems where the solution space is vast or poorly understood. These algorithms do not maintain a complete search tree but instead focus on moving towards a goal state based on local information.

Common Local Search Algorithms in AI:

Hill-Climbing Search Algorithm
Simulated Annealing
Local Beam Search
Genetic Algorithms
Tabu Search
3. Adversarial Search in AI
Adversarial search in AI refers to search algorithms designed to deal with competitive environments where multiple agents (often two) are in direct competition with one another, such as in games like chess, tic-tac-toe, or Go. These algorithms consider the actions of an opponent and attempt to minimize the possible losses in the worst-case scenario.

Here is a list of common adversarial search algorithms in AI:

Minimax Algorithm
Alpha-Beta Pruning
Negamax Algorithm
Expectiminimax Algorithm
Monte Carlo Tree Search (MCTS)
Zero-Sum Game Search
4. Constraint Satisfaction Problems
A Constraint Satisfaction Problem (CSP) is a problem-solving framework in Artificial intelligence. It involves variables, each with a domain of possible values, and constraints limiting the combinations of variable values. The objective is to find a consistent assignment satisfying all constraints. CSPs are widely used in scheduling, configuration, and optimization problems. Algorithms like backtracking and constraint propagation are employed to efficiently explore the solution space and find valid assignments.

Problem Structure in CSP’s
Constraint Propagation in CSP’s
Backtracking Search for CSP’s
Local Search for CSP’s
Knowledge, Reasoning and Planning in AI
Knowledge Representation in Artificial Intelligence
Knowledge representation in Artificial Intelligence (AI) refers to the way information, knowledge, and data are structured, stored, and used by AI systems to reason, learn, and make decisions. Effective knowledge representation is crucial for enabling machines to understand and interact with the world in a meaningful way.

Types of Knowledge

Declarative Knowledge: Knowledge that describes facts and concepts, often expressed in declarative sentences. It includes what someone knows about a subject.
Procedural Knowledge: This type involves knowing how to perform certain tasks or procedures, often expressed as rules or algorithms.
Meta-Knowledge: Knowledge about knowledge itself, including understanding the limits and capabilities of one's own knowledge.
Heuristic Knowledge: This represents rules of thumb or expert knowledge that guide problem-solving and decision-making.
Techniques for Knowledge Representation
Semantic Networks
Frames
Ontologies
Logical Representation
Production Rules
First Order Logic in Artificial Intelligence
First Order Logic (FOL), also known as Predicate Logic or First-Order Predicate Calculus, is a powerful formalism used in artificial intelligence (AI) to represent knowledge and reason about the world. Unlike propositional logic, which deals with simple, declarative propositions, FOL allows for the expression of more complex statements involving objects, their properties, and the relationships between them.

Knowledge Representation in First Order Logic
Syntax and Semantics of First Order Logic
Syntax
Terms and Formulas
Connectives and Quantifiers
Well-Formed Formulas (WFFs)
Semantics
Interpretations and Models
Truth Assignments
Satisfaction and Validity
Inference Rules in First Order Logic
Modus Ponens
Universal Instantiation
Existential Instantiation
Generalization Rules
Resolution in First Order Logic
Reasoning in Artificial Intelligence
Reasoning in Artificial Intelligence (AI) is the process by which AI systems draw conclusions, make decisions, or infer new knowledge from existing information. It is a core aspect of AI that enables machines to solve problems, understand environments, and make predictions based on logical thinking and learned patterns.

Types of Reasoning in AI

Deductive Reasoning
Inductive Reasoning
Abductive Reasoning
Analogical Reasoning
Common-Sense Reasoning
Planning in AI
Planning in AI refers to the process of generating a sequence of actions that an intelligent agent needs to execute to achieve specific goals or objectives. Planning is a critical aspect of artificial intelligence, particularly in areas such as robotics, autonomous systems, game AI, and complex problem-solving scenarios. Some of the planning techniques in artificial intelligence includes:

Classical Planning: Assumes a deterministic environment where actions have predictable outcomes. The planning problem is often represented using:
STRIPS (Stanford Research Institute Problem Solver)
PDDL (Planning Domain Definition Language)
Forward State Space Search
Probabilistic Planning: Deals with uncertainty in the environment, where actions may have probabilistic outcomes. The probabilistic planning techniques include:
Markov Decision Processes (MDPs)
Partially Observable Markov Decision Processes (POMDPs)
Bayesian Networks for Planning
Monte Carlo Tree Search (MCTS)
Hierarchical Planning: Breaks down complex tasks into simpler sub-tasks, often using a hierarchy of plans to solve different levels of the problem. The hierarchical planning include:
Hierarchical Task Networks (HTNs)
Hierarchical Reinforcement Learning (HRL)
Hierarchical State Space Search (HSSS)
Uncertain Knowledge and Reasoning in AI
Uncertain Knowledge and Reasoning in AI refers to the methods and techniques used to handle situations where information is incomplete, ambiguous, or uncertain. These approaches allow AI systems to make decisions, draw conclusions, or predict outcomes even when all the facts are not fully known or when data is noisy.

Types of Uncertainty
Aleatory Uncertainty: This is also known as "inherent uncertainty," and it stems from the randomness or variability inherent in the system or environment.
Epistemic Uncertainty: This arises from incomplete knowledge, lack of data, or imprecise measurements.
Techniques for Managing Uncertainty in AI
Dempster-Shafer Theory
Markov Decision Processes (MDPs)
Probabilistic Models
Fuzzy Logic
Belief Networks
Monte Carlo Methods
Learning in AI
Learning in Artificial Intelligence (AI) refers to the process by which a system improves its performance on a task over time through experience, data, or interaction with the environment. This is a core concept in AI, allowing systems to adapt to new situations, recognize patterns, make predictions, and make decisions without explicit programming for every possible scenario.

Supervised Learning
The AI system is trained on a labeled dataset, where the correct output is provided for each input. The goal is to learn a mapping from inputs to outputs that can generalize to new, unseen data.

Common Algorithms: Linear Regression, Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forests, Neural Networks.

Unsupervised Learning
The AI system is trained on an unlabeled dataset, where the correct output is not provided. The goal is to discover underlying patterns, structures, or relationships in the data.

Common Algorithms: K-Means Clustering, Hierarchical Clustering, Principal Component Analysis (PCA), Autoencoders.

Semi-Supervised Learning
Combines a small amount of labeled data with a large amount of unlabeled data during training. The goal is to improve learning performance using both types of data.

Common Algorithms: Semi-Supervised Support Vector Machines, Label Propagation, Co-training.

Reinforcement Learning
The AI system learns by interacting with an environment, receiving feedback in the form of rewards or punishments. The goal is to learn a policy that maximizes cumulative rewards over time.

Common Algorithms: Q-Learning, Deep Q-Networks (DQN), Policy Gradient Methods, SARSA (State-Action-Reward-State-Action).

Self-Supervised Learning
A type of unsupervised learning where the system generates its own labels from the input data. The model is trained to predict part of the input from other parts.

Common Algorithms: Contrastive Learning, Autoencoders, Generative Adversarial Networks (GANs).

Deep Learning
Deep Learning focuses on using neural networks with many layers (hence "deep") to model and understand complex patterns and representations in large datasets. It has revolutionized various fields by enabling machines to achieve human-like performance in tasks such as image recognition, natural language processing, and speech recognition.

Key Concepts in Deep Learning:

Artificial Neural Networks
Activation Functions
Recurrent Neural Network
Convolutional Neural Network
Probabilistic Models in AI
Probabilistic models in AI are a fundamental approach for dealing with uncertainty, making predictions, and modeling complex systems where uncertainty and variability play a crucial role. These models help in reasoning, decision-making, and learning from data.

Below is an overview of key probabilistic models and concepts in AI:

Bayesian Networks (Belief Networks)
Hidden Markov Models (HMMs)
Markov Decision Processes (MDPs)
Gaussian Mixture Models (GMMs)
Probabilistic Graphical Models (PGMs)
Naive Bayes Classifier
Variational Inference
Monte Carlo Methods
Expectation-Maximization (EM) Algorithm
Latent Variable Models
Communication, Perceiving, and Acting in AI and Robotics
Communication in AI and robotics involves the exchange of information between a machine and its environment, which includes humans, other machines, or even the surrounding physical world. This communication can take various forms, such as natural language processing, or through signals and sensors.

Perceiving is the process through which machines sense and interpret their surroundings. This involves the use of sensors, cameras, and other devices to gather data, which is then processed to create a meaningful representation of the environment.

Acting is the culmination of communication and perceiving, where a machine makes decisions and takes actions based on the information it has gathered and processed. In robotics, this could involve moving through an environment, manipulating objects, or even performing tasks autonomously.

Natural Language Processing (NLP)
NLP is a field of AI that enables machines to understand, interpret, and respond to human language in a valuable way. It combines computational linguistics with machine learning and deep learning to analyze text and speech.

Applications:

Chatbots and Virtual Assistants: NLP powers Siri, Alexa, and Google Assistant, allowing them to understand and respond to voice commands. Besides, chatbot-building software like BotSailor, ManyChat, ChatPion, Watti, Chatfuel, BotSify, etc. use NLP to understand customer chat and response to text commands.
Sentiment Analysis: Used in social media monitoring, NLP helps companies understand customer opinions and emotions from text data.
Translation Services: Tools like Google Translate use NLP to translate text from one language to another.
Text Generation: Models like GPT-4 can generate coherent and contextually relevant text, assisting in content creation and conversational AI.
Computer Vision
Computer Vision is a branch of AI that enables machines to interpret and make decisions based on visual data. It involves techniques for acquiring, processing, and analyzing images and videos.

Applications:

Facial Recognition: Used in security systems, social media tagging, and even payment systems, facial recognition identifies and verifies individuals from images or video.
Autonomous Vehicles: Computer Vision is critical in self-driving cars for detecting and interpreting traffic signals, pedestrians, and other vehicles.
Medical Imaging: In healthcare, it helps in diagnosing diseases by analyzing medical images like X-rays and MRIs.
Image and Video Analysis: Applications range from surveillance systems to social media platforms where visual content is automatically categorized and analyzed.
Robotics
Robotics is a field of AI that deals with the design, construction, operation, and application of robots. It often integrates both NLP and Computer Vision to enable robots to interact with their environment and perform tasks autonomously.

Applications:

Industrial Automation: Robots are used in manufacturing for tasks like assembly, painting, and welding, which require precision and consistency.
Service Robots: These include robots used in healthcare (e.g., surgery-assisting robots), customer service, and hospitality.
Autonomous Robots: From vacuum cleaners like Roomba to delivery robots, these machines can navigate and perform tasks in dynamic environments with minimal human intervention.
Humanoid Robots: Advanced robots designed to mimic human actions and interact with humans, such as ASIMO by Honda.
Generative AI
1. Generative Adversarial Networks (GANs)
GANs consist of two neural networks, a generator and a discriminator, that compete in a zero-sum game. The generator tries to create data that is indistinguishable from real data, while the discriminator attempts to distinguish between real and generated data.

GANs are widely used in image generation, style transfer, video synthesis, and creating realistic data for training models. Examples include generating photorealistic images, deepfakes, and enhancing image resolution.

2. Variational Autoencoders (VAEs)
VAEs are generative models that encode input data into a latent space, from which new data samples can be generated. The model is trained to maximize the likelihood of the data while regularizing the latent space to follow a predefined distribution, typically Gaussian.

VAEs are used in image generation, anomaly detection, and data compression. They are particularly effective in generating variations of input data, such as creating new faces based on an existing dataset.

3. Diffusion Models
Diffusion models generate data by simulating the process of gradually transforming noise into structured data through a series of steps. They reverse a diffusion process, starting from a noise distribution and progressively refining the data.

Diffusion models are increasingly popular in image synthesis and have shown promise in generating high-quality images with fine details. They are also used in fields like speech synthesis and molecular generation.

4. Transformers
Transformers are a type of deep learning model that rely on self-attention mechanisms to process and generate sequential data, such as text or time series. They excel at understanding context and relationships between data points across long sequences.

Transformers are the backbone of many state-of-the-art models in natural language processing, such as GPT-3 and BERT, enabling tasks like text generation, translation, summarization, and more. They are also applied in areas like protein folding and image generation.

5. Autoregressive Models
Autoregressive models generate data by predicting each element based on the previously generated elements. The process is sequential, meaning the model generates one data point at a time, conditioned on past outputs.

These models are extensively used in text generation (e.g., GPT models), time series prediction, and audio generation. They are particularly effective in tasks that require coherent sequential output, such as generating text that makes logical sense or predicting future values in a series.

6. Flow-based Models
Flow-based models learn to generate data by applying a sequence of invertible transformations to a simple distribution (like a Gaussian). This allows for exact likelihood estimation and efficient sampling, making them unique among generative models.

Flow-based models are used in image generation, density estimation, and anomaly detection. They offer the advantage of providing exact probability calculations for generated data, which is useful in tasks requiring precise control over the data generation process.
