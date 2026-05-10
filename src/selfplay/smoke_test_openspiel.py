import pyspiel


game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

while not state.is_terminal():
    legal = state.legal_actions()
    action = legal[0]
    state.apply_action(action)

print("Returns:", state.returns())
