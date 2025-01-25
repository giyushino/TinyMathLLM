# TinyMathLLM

## Purpose
This project aims to determine whether or not a large language model with "only" 1.1 billion parameters can learn math with minimal compute power. LLMs have become increasingly proficient in all fields, including math. With improvements in chain of thought, training strategies, etc, LLMs are now more capable than ever, but they are also becoming increasingly harder to run locally as they naturally grow larger in size.

The idea I would like to test with this repo is if small, lightweight models can become capable math chatbots. Currently, my hypothesis is yes, but I believe that model would need to act more as an intermediary between the user and a math API. Rather than having the model attempt to laern to solve all kinds of different problems, I believe it would be much easier for the model to learn to extract information from user inputs, determine what kind of question it is, and then use the proper tools to solve accordingly. 

Currently, I'm only testing the model on questions regarding addition, multiplication, derivatives, integration, matrix multiplication, and determinants. Using 4-bit quantization and low rank adaptation, I'm making training and inference extremely light, though I believe this is having a non-neglible impact on model perfomance. Once I get access to more compute power, I will train the model on answering questions + extracting information/classification again. 




