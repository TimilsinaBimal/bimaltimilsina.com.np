---
title: "How I used AWS to Serve Open Sourse LLM to thousands of users"
date: 2024-08-27
tags: ["Large Language Models", "Finetuning"]
author: "Bimal Timilsina"
description: "LoRA is reshaping how we fine-tune neural networks, offering lightning-fast training, reduced memory footprint, and impressive performance gains. With LoRA, achieving optimal model performance is not just efficient—it's a game-changer for AI development."
categories: ["Large Language Models"]
thumbnail: lora.jpg
cover: lora.jpg
type: post
---
## Introduction

From the past few months, I was experimenting with different open source Large Language Models. As a result of that, we recently moved from OpenAI GPT-4 to Llama3.1 model. We have very different requirements and choosing open-source LLM gave us control over things that we wanted. We could tweak or finetune model based on our data and modify according to our needs. We experimented with different LLMs such as Mistral, Gemma but finally landed into Llama3.1 as it felt good for our use case. I will talk about it in some other articles, but let's move our focus on the today's main topic. Scaling.
While using open source models, we had different choices. We could use APIs that different third-party platforms hosted or we also had a choice to host the model on our own. And we chose the second approach. Yes, it may have been cheaper if we used hosted APIs but we wanted more control over the choices. So, we moved into second approach of hosting model by ourself.

But, Then came the problems and problems. There were so many things we needed to consider before hosting these models. We needed to design our own infrastructure find the optimal servers and GPU requirements and the issues with latency, server maintainance etc.

But wait, we wanted to go one by one and that's what we did. Let's dive them together one by one.

### Choosing the Infrastructure
We did not have many options for this. The budget was too much for us to buy GPUs and maintain in-house servers. So, we opted for AWS here. Also, because we had few credits left and it seemed the right choice. But, that came with it's own set of issues. We needed to find which service to use. We could use EKS, Sagemaker or just EC2 instances. I think many people prefer EKS, but I didn't have much knowledge on EKS and had very little time to explore since we were near our launch time. So, I chose EC2 instances. To be specific, I went with `g5` instance which as enough for us to run Llama3.1 8B model with int8 quantization. We also checked `g4dn` but it was a bit slow for our use case and other too much expensive for our traffic and use cases. I will talk about setup in details in later sections. But for now, let's move forward.

### Choosing the Inference Server
