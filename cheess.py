import sys
import chess
import requests
import json
from typing import Optional, List, Dict
import time
from enum import Enum
import random

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QMessageBox, QGroupBox, QTextEdit, QGridLayout,
                             QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush

class GameMode(Enum):
    HUMAN_VS_AI = 1
    AI_VS_AI = 2

class PlayerType(Enum):
    HUMAN = "Human"
    AI = "AI"

class ChessBoardWidget(QWidget):
    square_clicked = pyqtSignal(int)  # emits square index (0-63)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.square_size = 50
        self.selected_square = None
        self.valid_moves = []
        self.board = None
        self.game_over = False
        
        # Colors
        self.light_square = QColor(240, 217, 181)
        self.dark_square = QColor(181, 136, 99)
        self.highlight_color = QColor(100, 249, 83, 150)
        self.check_color = QColor(255, 0, 0, 100)
        self.game_over_color = QColor(255, 255, 0, 100)
        
    def set_board(self, board, game_over=False):
        self.board = board
        self.game_over = game_over
        self.update()
        
    def set_selection(self, square, valid_moves):
        if not self.game_over:
            self.selected_square = square
            self.valid_moves = valid_moves
            self.update()
        
    def clear_selection(self):
        self.selected_square = None
        self.valid_moves = []
        self.update()

    def paintEvent(self, event):
        if not self.board:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate square size to fit the widget
        size = min(self.width(), self.height())
        self.square_size = size // 8
        offset_x = (self.width() - self.square_size * 8) // 2
        offset_y = (self.height() - self.square_size * 8) // 2
        
        # Draw board
        for row in range(8):
            for col in range(8):
                color = self.light_square if (row + col) % 2 == 0 else self.dark_square
                rect = (offset_x + col * self.square_size, 
                       offset_y + row * self.square_size, 
                       self.square_size, self.square_size)
                
                painter.fillRect(*rect, color)
                
                # Highlight selected square
                square_idx = chess.square(col, 7 - row)
                if self.selected_square == square_idx:
                    painter.fillRect(*rect, self.highlight_color)
                
                # Highlight valid moves
                for move in self.valid_moves:
                    if move.from_square == self.selected_square and move.to_square == square_idx:
                        center_x = offset_x + col * self.square_size + self.square_size // 2
                        center_y = offset_y + row * self.square_size + self.square_size // 2
                        radius = self.square_size // 6
                        painter.setBrush(QBrush(self.highlight_color))
                        painter.setPen(Qt.NoPen)
                        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                self.draw_piece(painter, square, piece, offset_x, offset_y)
        
        # Highlight king in check
        if self.board.is_check():
            king_color = self.board.turn
            king_square = self.board.king(king_color)
            if king_square is not None:
                col = chess.square_file(king_square)
                row = 7 - chess.square_rank(king_square)
                rect = (offset_x + col * self.square_size, 
                       offset_y + row * self.square_size, 
                       self.square_size, self.square_size)
                painter.fillRect(*rect, self.check_color)
        
        # Draw coordinates
        painter.setPen(QColor(0, 0, 0))
        font = QFont("Arial", 10)
        painter.setFont(font)
        for i in range(8):
            # File letters (a-h)
            painter.drawText(offset_x + i * self.square_size + self.square_size - 15, 
                           offset_y + 8 * self.square_size - 5, chr(97 + i))
            # Rank numbers (1-8)
            painter.drawText(offset_x - 15, 
                           offset_y + i * self.square_size + self.square_size - 5, str(8 - i))
        
        # Draw game over overlay
        if self.game_over:
            painter.fillRect(offset_x, offset_y, self.square_size * 8, self.square_size * 8, 
                           self.game_over_color)
            painter.setPen(QColor(0, 0, 0))
            font = QFont("Arial", 20, QFont.Bold)
            painter.setFont(font)
            painter.drawText(offset_x, offset_y, self.square_size * 8, self.square_size * 8,
                           Qt.AlignCenter, "GAME OVER")

    def draw_piece(self, painter, square, piece, offset_x, offset_y):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)  # Flip vertically
        
        x = offset_x + col * self.square_size
        y = offset_y + row * self.square_size
        
        # Choose color
        color = Qt.white if piece.color == chess.WHITE else Qt.black
        bg_color = Qt.black if piece.color == chess.WHITE else Qt.white
        
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(bg_color))
        
        # Draw piece background
        margin = self.square_size // 6
        painter.drawEllipse(x + margin, y + margin, 
                          self.square_size - 2 * margin, self.square_size - 2 * margin)
        
        # Draw piece symbol
        font = QFont("Arial", self.square_size // 2)
        painter.setFont(font)
        symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        symbol = symbols.get(piece.symbol(), '?')
        
        painter.drawText(x, y, self.square_size, self.square_size, 
                        Qt.AlignCenter, symbol)

    def mousePressEvent(self, event):
        if not self.board or self.game_over:
            return
            
        x = event.x()
        y = event.y()
        
        size = min(self.width(), self.height())
        square_size = size // 8
        offset_x = (self.width() - square_size * 8) // 2
        offset_y = (self.height() - square_size * 8) // 2
        
        if (offset_x <= x < offset_x + square_size * 8 and 
            offset_y <= y < offset_y + square_size * 8):
            col = (x - offset_x) // square_size
            row = (y - offset_y) // square_size
            square = chess.square(col, 7 - row)  # Flip row for chess coordinates
            self.square_clicked.emit(square)

class AIMoveWorker(QThread):
    move_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ollama_url, model, board):
        super().__init__()
        self.ollama_url = ollama_url
        self.model = model
        self.board = board
        self.prompt = self.create_prompt()

    def create_prompt(self):
        legal_moves_san = [self.board.san(move) for move in self.board.legal_moves]
        return f"""
You are a chess engine. Analyze the current chess position and suggest the best move.

Current board state (White at bottom, Black at top):
{self.board}

FEN: {self.board.fen()}

Legal moves: {legal_moves_san}

It is {'white' if self.board.turn else 'black'}'s turn.

IMPORTANT: Respond with ONLY the move in Standard Algebraic Notation (SAN)
Choose from these legal moves: {', '.join(legal_moves_san)}

Examples: e4, Nf3, O-O, Qxd5, exd5

Your move (SAN format only):
"""

    def extract_move_from_response(self, response):
        if not response:
            return ""
            
        response = response.strip()
        response = response.replace('"', '').replace("'", "")
        
        # Remove common prefixes
        prefixes = [
            "i would play", "my move is", "the best move is", "move:", 
            "suggested move:", "let's play", "i choose", "move", "response:",
            "the move is", "i suggest", "i recommend"
        ]
        
        for prefix in prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip(' :.-')
        
        # Take only the first line and remove any trailing punctuation
        response = response.split('.')[0].split('\n')[0].strip()
        
        # Try to find a valid SAN move in the response
        words = response.split()
        for word in words:
            word = word.strip('.,!?;:()[]{}')
            # Basic SAN pattern matching
            if (any(c.isalpha() for c in word) and any(c.isdigit() for c in word) or
                word.upper() in ['O-O', 'O-O-O'] or
                len(word) in [2, 3, 4] and word[0].isalpha() and word[1].isdigit()):
                return word
        
        # If no pattern match, return the first alphanumeric word
        for word in words:
            word = word.strip('.,!?;:()[]{}')
            if any(c.isalnum() for c in word):
                return word
        
        return ""

    def run(self):
        try:
            # First, check if Ollama is available
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code != 200:
                raise Exception("Ollama server not responding")
            
            payload = {
                "model": self.model,
                "prompt": self.prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "num_predict": 100
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Increased timeout
            )
            response.raise_for_status()
            
            result = response.json()
            move_str = self.extract_move_from_response(result["response"])
            
            if move_str:
                self.move_ready.emit(move_str)
            else:
                raise Exception("AI returned empty or invalid move")
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class ChessGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ollama_url = "http://localhost:11434"
        self.board = chess.Board()
        self.game_mode = None
        self.white_player = None
        self.black_player = None
        self.white_model = None
        self.black_model = None
        self.available_models = []
        self.ai_worker = None
        self.game_active = False
        self.ai_thinking = False
        
        self.init_ui()
        self.load_available_models()
        
    def init_ui(self):
        self.setWindowTitle("Chess with Ollama - AI vs Human / AI vs AI")
        self.setGeometry(100, 100, 900, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Chess board
        left_panel = QVBoxLayout()
        
        self.chess_board = ChessBoardWidget()
        self.chess_board.set_board(self.board)
        self.chess_board.square_clicked.connect(self.handle_square_click)
        
        left_panel.addWidget(self.chess_board)
        
        # Right panel - Controls and info
        right_panel = QVBoxLayout()
        
        # Game mode selection
        mode_group = QGroupBox("Game Setup")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human vs AI", "AI vs AI"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(QLabel("Game Mode:"))
        mode_layout.addWidget(self.mode_combo)
        
        # Color selection for Human vs AI
        self.color_group = QGroupBox("Human Color (Human vs AI)")
        color_layout = QHBoxLayout()
        self.white_radio = QPushButton("Play as White")
        self.black_radio = QPushButton("Play as Black")
        self.white_radio.setCheckable(True)
        self.black_radio.setCheckable(True)
        self.white_radio.setChecked(True)
        self.white_radio.clicked.connect(self.on_color_changed)
        self.black_radio.clicked.connect(self.on_color_changed)
        color_layout.addWidget(self.white_radio)
        color_layout.addWidget(self.black_radio)
        self.color_group.setLayout(color_layout)
        mode_layout.addWidget(self.color_group)
        
        # Model selection
        model_layout = QHBoxLayout()
        
        self.white_model_combo = QComboBox()
        self.black_model_combo = QComboBox()
        
        model_layout.addWidget(QLabel("White AI:"))
        model_layout.addWidget(self.white_model_combo)
        model_layout.addWidget(QLabel("Black AI:"))
        model_layout.addWidget(self.black_model_combo)
        
        mode_layout.addLayout(model_layout)
        
        # Start game button
        self.start_btn = QPushButton("Start Game")
        self.start_btn.clicked.connect(self.start_game)
        mode_layout.addWidget(self.start_btn)
        
        mode_group.setLayout(mode_layout)
        right_panel.addWidget(mode_group)
        
        # Game info
        info_group = QGroupBox("Game Information")
        info_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Ready to start")
        self.turn_label = QLabel("Turn: White")
        self.move_label = QLabel("Move: 1")
        self.fen_label = QLabel("FEN: ")
        self.game_result_label = QLabel("Result: ")
        
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.turn_label)
        info_layout.addWidget(self.move_label)
        info_layout.addWidget(self.fen_label)
        info_layout.addWidget(self.game_result_label)
        
        info_group.setLayout(info_layout)
        right_panel.addWidget(info_group)
        
        # Move history
        history_group = QGroupBox("Move History")
        history_layout = QVBoxLayout()
        
        self.move_history = QTextEdit()
        self.move_history.setMaximumHeight(150)
        self.move_history.setReadOnly(True)
        history_layout.addWidget(self.move_history)
        
        history_group.setLayout(history_layout)
        right_panel.addWidget(history_group)
        
        # Controls
        control_group = QGroupBox("Game Controls")
        control_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("Undo Move")
        self.undo_btn.clicked.connect(self.undo_move)
        self.reset_btn = QPushButton("Reset Game")
        self.reset_btn.clicked.connect(self.reset_game)
        
        control_layout.addWidget(self.undo_btn)
        control_layout.addWidget(self.reset_btn)
        
        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)
        
        # AI thinking indicator
        self.thinking_bar = QProgressBar()
        self.thinking_bar.setVisible(False)
        self.thinking_bar.setRange(0, 0)  # Indeterminate progress
        right_panel.addWidget(self.thinking_bar)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        self.update_display()
        
    def on_mode_changed(self, index):
        # Show/hide color selection based on mode
        if index == 0:  # Human vs AI
            self.color_group.setVisible(True)
        else:  # AI vs AI
            self.color_group.setVisible(False)
            
    def on_color_changed(self):
        # Ensure only one color is selected
        if self.sender() == self.white_radio and self.white_radio.isChecked():
            self.black_radio.setChecked(False)
        elif self.sender() == self.black_radio and self.black_radio.isChecked():
            self.white_radio.setChecked(False)
        
    def load_available_models(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model["name"] for model in data.get("models", [])]
                
                self.white_model_combo.clear()
                self.black_model_combo.clear()
                
                for model in self.available_models:
                    self.white_model_combo.addItem(model)
                    self.black_model_combo.addItem(model)
                    
                if self.available_models:
                    self.white_model_combo.setCurrentIndex(0)
                    self.black_model_combo.setCurrentIndex(0)
                    self.status_label.setText("Status: Models loaded successfully")
                else:
                    self.status_label.setText("Status: No models found")
            else:
                self.status_label.setText("Status: Could not connect to Ollama")
                QMessageBox.warning(self, "Connection Error", 
                                  "Could not connect to Ollama. AI modes will not be available.")
        except Exception as e:
            self.status_label.setText(f"Status: Ollama connection failed")
            QMessageBox.warning(self, "Connection Error", 
                              f"Could not connect to Ollama: {str(e)}\nAI modes will not be available.")

    def start_game(self):
        mode_index = self.mode_combo.currentIndex()
        
        if mode_index == 0:  # Human vs AI
            if not self.available_models:
                QMessageBox.warning(self, "No Models", "No Ollama models available. Please install models first.")
                return
                
            # Set players based on color selection
            if self.white_radio.isChecked():
                self.white_player = PlayerType.HUMAN
                self.black_player = PlayerType.AI
                self.black_model = self.black_model_combo.currentText()
            else:
                self.white_player = PlayerType.AI
                self.black_player = PlayerType.HUMAN
                self.white_model = self.white_model_combo.currentText()
                
            self.game_mode = GameMode.HUMAN_VS_AI
            
        elif mode_index == 1:  # AI vs AI
            if not self.available_models:
                QMessageBox.warning(self, "No Models", "No Ollama models available. Please install models first.")
                return
                
            self.white_player = PlayerType.AI
            self.black_player = PlayerType.AI
            self.white_model = self.white_model_combo.currentText()
            self.black_model = self.black_model_combo.currentText()
            self.game_mode = GameMode.AI_VS_AI
        
        self.game_active = True
        self.ai_thinking = False
        self.reset_game()
        self.update_display()
        self.add_to_history("Game started!")
        
        # If AI plays first, start AI move
        if self.game_active and (self.game_mode == GameMode.AI_VS_AI or (
            self.game_mode == GameMode.HUMAN_VS_AI and self.white_player == PlayerType.AI)):
            QTimer.singleShot(500, self.make_ai_move)  # Small delay to show UI update

    def handle_square_click(self, square):
        if self.game_mode is None or not self.game_active or self.ai_thinking:
            return
            
        current_player = self.white_player if self.board.turn else self.black_player
        
        if current_player != PlayerType.HUMAN:
            return
            
        piece = self.board.piece_at(square)
        
        # If no square selected yet, select if it's the player's piece
        if self.chess_board.selected_square is None:
            if piece and ((piece.color == chess.WHITE and self.white_player == PlayerType.HUMAN) or 
                         (piece.color == chess.BLACK and self.black_player == PlayerType.HUMAN)):
                valid_moves = [move for move in self.board.legal_moves if move.from_square == square]
                self.chess_board.set_selection(square, valid_moves)
        else:
            # Try to make a move
            from_square = self.chess_board.selected_square
            move = None
            
            # Find the move in legal moves
            for legal_move in self.board.legal_moves:
                if legal_move.from_square == from_square and legal_move.to_square == square:
                    move = legal_move
                    break
            
            # Handle pawn promotion
            if move and self.board.piece_at(from_square).piece_type == chess.PAWN:
                if chess.square_rank(square) in [0, 7]:  # Promotion rank
                    # Auto-promote to queen for simplicity
                    move = chess.Move(from_square, square, promotion=chess.QUEEN)
            
            if move and move in self.board.legal_moves:
                self.make_move(move)
            else:
                # Select new piece if it's the player's color
                if piece and ((piece.color == chess.WHITE and self.white_player == PlayerType.HUMAN) or 
                             (piece.color == chess.BLACK and self.black_player == PlayerType.HUMAN)):
                    valid_moves = [move for move in self.board.legal_moves if move.from_square == square]
                    self.chess_board.set_selection(square, valid_moves)
                else:
                    self.chess_board.clear_selection()

    def make_move(self, move):
        if not self.game_active:
            return
            
        san_move = self.board.san(move)
        self.board.push(move)
        
        player_name = "White" if not self.board.turn else "Black"  # Opposite because turn changed
        player_type = self.white_player if player_name == "White" else self.black_player
        
        if player_type == PlayerType.AI:
            model = self.white_model if player_name == "White" else self.black_model
            move_text = f"{player_name} ({model}): {san_move}"
        else:
            move_text = f"{player_name} (Human): {san_move}"
            
        self.add_to_history(move_text)
        self.update_display()
        self.chess_board.clear_selection()
        
        # Check game status
        if self.board.is_game_over():
            self.handle_game_over()
        elif self.game_active:
            # If it's AI's turn, make AI move
            current_player = self.white_player if self.board.turn else self.black_player
            if current_player == PlayerType.AI:
                QTimer.singleShot(500, self.make_ai_move)  # Small delay for better UX

    def make_ai_move(self):
        if not self.game_active or self.ai_thinking:
            return
            
        if not self.board.legal_moves:
            self.handle_game_over()
            return
            
        current_player = "White" if self.board.turn else "Black"
        model = self.white_model if self.board.turn else self.black_model
        
        self.ai_thinking = True
        self.thinking_bar.setVisible(True)
        self.status_label.setText(f"Status: {current_player} AI ({model}) is thinking...")
        
        self.ai_worker = AIMoveWorker(self.ollama_url, model, self.board)
        self.ai_worker.move_ready.connect(self.handle_ai_move)
        self.ai_worker.error_occurred.connect(self.handle_ai_error)
        self.ai_worker.start()

    def handle_ai_move(self, move_str):
        self.ai_thinking = False
        self.thinking_bar.setVisible(False)
        
        if not self.game_active:
            return
            
        if not move_str:
            self.status_label.setText("Status: AI returned empty move, using fallback")
            self.use_fallback_move()
            return
            
        # Validate and make the move
        try:
            move = self.board.parse_san(move_str)
            if move in self.board.legal_moves:
                self.make_move(move)
                return
        except:
            pass  # Try other formats
            
        # Try UCI notation
        try:
            move = chess.Move.from_uci(move_str.lower())
            if move in self.board.legal_moves:
                self.make_move(move)
                return
        except:
            pass  # Try other approaches
            
        # If we get here, the move is invalid
        self.status_label.setText("Status: AI suggested invalid move, using fallback")
        self.use_fallback_move()

    def use_fallback_move(self):
        """Use a fallback move when AI fails"""
        if self.board.legal_moves:
            # Try to find a sensible move first
            sensible_moves = []
            for move in self.board.legal_moves:
                # Prefer captures
                if self.board.is_capture(move):
                    sensible_moves.append(move)
                # Prefer checks
                board_copy = self.board.copy()
                board_copy.push(move)
                if board_copy.is_check():
                    sensible_moves.append(move)
            
            if sensible_moves:
                fallback_move = random.choice(sensible_moves)
            else:
                fallback_move = random.choice(list(self.board.legal_moves))
                
            self.make_move(fallback_move)
        else:
            self.handle_game_over()

    def handle_ai_error(self, error_msg):
        self.ai_thinking = False
        self.thinking_bar.setVisible(False)
        self.status_label.setText(f"Status: AI error - {error_msg}")
        
        # Use fallback move
        self.use_fallback_move()

    def handle_game_over(self):
        self.game_active = False
        self.ai_thinking = False
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            message = f"Checkmate! {winner} wins!"
            result_text = f"Result: {winner} wins by checkmate"
        elif self.board.is_stalemate():
            message = "Stalemate! Game is a draw."
            result_text = "Result: Draw by stalemate"
        elif self.board.is_insufficient_material():
            message = "Draw by insufficient material."
            result_text = "Result: Draw by insufficient material"
        elif self.board.is_fifty_moves():
            message = "Draw by fifty-move rule."
            result_text = "Result: Draw by fifty-move rule"
        elif self.board.is_repetition():
            message = "Draw by threefold repetition."
            result_text = "Result: Draw by threefold repetition"
        else:
            message = "Game over!"
            result_text = "Result: Game over"
            
        self.status_label.setText(f"Status: {message}")
        self.game_result_label.setText(result_text)
        self.add_to_history(f"*** {message} ***")
        
        # Update the board to show game over state
        self.chess_board.set_board(self.board, game_over=True)
        
        QMessageBox.information(self, "Game Over", message)

    def undo_move(self):
        if len(self.board.move_stack) > 0 and self.game_active and not self.ai_thinking:
            self.board.pop()
            self.update_display()
            self.add_to_history("Last move undone")
            
            # Clear any selection
            self.chess_board.clear_selection()

    def reset_game(self):
        self.board = chess.Board()
        self.game_active = True
        self.ai_thinking = False
        self.chess_board.set_board(self.board, game_over=False)
        self.chess_board.clear_selection()
        self.move_history.clear()
        self.game_result_label.setText("Result: ")
        self.thinking_bar.setVisible(False)
        
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait()
            
        self.update_display()

    def update_display(self):
        self.chess_board.set_board(self.board, game_over=not self.game_active)
        
        # Update labels
        turn = "White" if self.board.turn else "Black"
        self.turn_label.setText(f"Turn: {turn}")
        self.move_label.setText(f"Move: {self.board.fullmove_number}")
        self.fen_label.setText(f"FEN: {self.board.fen()}")
        
        # Update status
        if not self.game_active:
            # Status already set by game over handler
            pass
        elif self.ai_thinking:
            # Don't change status during AI thinking
            pass
        elif self.board.is_check():
            self.status_label.setText("Status: CHECK!")
        elif self.board.is_game_over():
            self.handle_game_over()
        else:
            current_player = self.white_player if self.board.turn else self.black_player
            if current_player == PlayerType.AI and self.game_mode == GameMode.AI_VS_AI:
                model = self.white_model if self.board.turn else self.black_model
                self.status_label.setText(f"Status: {turn} AI ({model}) thinking...")
            elif current_player == PlayerType.AI:
                model = self.white_model if self.board.turn else self.black_model
                self.status_label.setText(f"Status: {turn} AI ({model}) thinking...")
            else:
                self.status_label.setText(f"Status: {turn} to move")

    def add_to_history(self, text):
        current_text = self.move_history.toPlainText()
        if current_text:
            self.move_history.setPlainText(current_text + "\n" + text)
        else:
            self.move_history.setPlainText(text)
        
        # Scroll to bottom
        self.move_history.verticalScrollBar().setValue(
            self.move_history.verticalScrollBar().maximum()
        )

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = ChessGame()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()