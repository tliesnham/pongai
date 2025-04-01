# AI Pong Trainer

[![PyGame](https://img.shields.io/badge/PyGame-2.5.2-blue)](https://www.pygame.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)

A classic Pong implementation with AI training capabilities, featuring telemetry logging and neural network integration.

![Pong Gameplay Demo](demo.gif) *Add demo GIF later*

## Project Overview
This repository contains a PyGame-based Pong game that:
- Logs comprehensive gameplay telemetry for machine learning
- Implements heuristic-based AI for baseline performance
- Provides infrastructure for training a PyTorch neural network
- Enables comparative analysis between different AI approaches

## Key Features
- 🏓 Classic Pong gameplay mechanics
- 📊 Real-time telemetry logging (position, velocity, actions)
- 🤖 Configurable heuristic AI opponent
- 🧠 PyTorch model integration (WIP)
- 📈 CSV data pipeline for training
- 🆚 AI vs AI comparison framework

## Installation
```bash
git clone https://github.com/tliesnham/pongai.git
cd pongai
pip install -r requirements.txt