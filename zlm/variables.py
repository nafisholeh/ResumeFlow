'''
-----------------------------------------------------------------------
File: zlm/variables.py
Creation Time: Aug 18th 2024, 5:26 am
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''

from zlm.prompts.sections_prompt import EXPERIENCE, SKILLS, PROJECTS, EDUCATIONS, CERTIFICATIONS, ACHIEVEMENTS
from zlm.schemas.sections_schemas import Achievements, Certifications, Educations, Experiences, Projects, SkillSections

GPT_EMBEDDING_MODEL = "text-embedding-ada-002"
# text-embedding-3-large, text-embedding-3-small

GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
# models/embedding-001

OLLAMA_EMBEDDING_MODEL = "bge-m3"

DEFAULT_LLM_PROVIDER = "Gemini"
DEFAULT_LLM_MODEL = "gemini-1.5-flash"

LLM_MAPPING = {
    'GPT': {
        "api_env": "OPENAI_API_KEY",
        "model": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-1106-preview", "gpt-3.5-turbo"],
    },
    'Gemini': {
        "api_env": "GEMINI_API_KEY",
        "model": ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-1.5-pro-latest", "gemini-1.5-pro-exp-0801"], # "gemini-1.0-pro", "gemini-1.0-pro-latest"
    },
    'OpenRouter': {
        "api_env": "OPENROUTER_API_KEY",
        "model": [
            # Free models (look for :free suffix)
            "google/gemma-3-27b-it:free",      # Google's Gemma 3 27B - Great for instructions and text generation
            "qwen/qwen3-30b-a3b:free",         # Qwen3 30B A3B - High quality general purpose model
            "qwen/qwen3-8b:free",              # Qwen3 8B - Smaller, faster model
            "deepseek/deepseek-v3-0324:free",  # DeepSeek V3 - Powerful 685B parameter model
            "mistralai/mistral-7b-instruct:free",  # Mistral 7B Instruct - Good for instructions and coding
            "microsoft/phi-3-mini-4k-instruct:free",  # Phi-3 Mini 4K - Microsoft's compact model
            "01-ai/yi-1.5-34b:free",           # Yi 1.5 34B - Strong general purpose model

            # Paid models - Best for resume generation
            "anthropic/claude-3.5-sonnet",     # Claude 3.5 Sonnet - Excellent for creative writing and summarization
            "anthropic/claude-3-opus",         # Claude 3 Opus - Anthropic's most powerful model
            "anthropic/claude-3-sonnet",       # Claude 3 Sonnet - Good balance of quality and cost
            "anthropic/claude-3-haiku",        # Claude 3 Haiku - Faster, more affordable Claude model
            "google/gemini-1.5-pro",           # Gemini 1.5 Pro - Google's advanced model
            "google/gemini-1.5-flash",         # Gemini 1.5 Flash - Faster Google model
            "openai/gpt-4o",                   # GPT-4o - OpenAI's latest model
            "openai/gpt-4o-mini",              # GPT-4o Mini - Smaller, faster GPT-4o
            "meta-llama/llama-3-70b-instruct", # Llama 3 70B - Meta's largest model
            "meta-llama/llama-3-8b-instruct",  # Llama 3 8B - Smaller Meta model
        ],
    },
    # 'Ollama': {
    #     "api_env": None,
    #     "model": ['llama3.1', 'llama3'],
    # }
}

section_mapping = {
    "work_experience": {"prompt":EXPERIENCE, "schema": Experiences},
    "skill_section": {"prompt":SKILLS, "schema": SkillSections},
    "projects": {"prompt":PROJECTS, "schema": Projects},
    "education": {"prompt":EDUCATIONS, "schema": Educations},
    "certifications": {"prompt":CERTIFICATIONS, "schema": Certifications},
    "achievements": {"prompt":ACHIEVEMENTS, "schema": Achievements},
}