# Chess with Ollama AI

 Play against AI or watch AI vs AI matches with a PyQt5 interface.
## Demo

![Chess with Ollama Demo](2025-10-16-19-14-31_edited.gif)


## Features

### 🎮 Game Modes
- **Human vs AI**: Play against AI with your choice of color
- **AI vs AI**: Watch two AI models play against each other

### 🤖 AI Integration
- **Ollama Support**: Uses locally running Ollama models for chess moves
- **Multiple Models**: Choose from any available Ollama models
- **Smart Fallback**: Automatic fallback to random legal moves if AI fails

### 🎨 User Interface
- **Visual Chess Board**: Clean, modern chess board with piece symbols
- **Move Highlights**: Visual indicators for selected pieces and valid moves
- **Game Status**: Real-time game information and move history
- **Check Detection**: Visual highlighting when king is in check

### ⚙️ Game Features
- **Standard Chess Rules**: Full chess rule implementation
- **Move History**: Complete record of all moves
- **Undo Function**: Undo last move during gameplay
- **Game Over Detection**: Automatic detection of checkmate, stalemate, and draws
- **FEN Notation**: Display current position in FEN format

## Installation

### Prerequisites

1. **Python 3.7+** - [Download Python](https://www.python.org/downloads/)
2. **Ollama** - [Install Ollama](https://ollama.ai/)
3. **Chess Models** - Install at least two model in Ollama

### Install Dependencies

```bash
pip install PyQt5 chess requests
