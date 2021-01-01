package put.ai.games.bestplayer;

import java.util.Comparator;
import java.util.List;
import java.util.Random;

import put.ai.games.game.Board;
import put.ai.games.game.Move;
import put.ai.games.game.Player;

public class BestPlayer extends Player {

  private final Algorithm algorithm = new ChooseBestAlgorithm(new RandomHeuristic());

  @Override
  public String getName() {
    return "Zuzanna Juszczak 141238 Stanisław Kaczmarek 141240";
  }

  @Override
  public Move nextMove(Board board) {
    return algorithm.nextMove(getColor(), board, getTime());
  }

  /**
   * Interface of heuristic for game state evaluation
   */
  private interface Heuristic {

    /**
     * Evalulate board for given player
     *
     * @param color color of current player
     * @param board game board
     * @return score as double (higher = better)
     */
    double getScore(Color color, Board board);
  }

  /**
   * Random score
   */
  private static class RandomHeuristic implements Heuristic {

    private static final Random random = new Random(0xdeadbeef);

    @Override
    public double getScore(Color color, Board board) {
      return random.nextDouble();
    }
  }

  /**
   * Interface of algorithm used to determine next move
   */
  private interface Algorithm {

    /**
     * Get best next move
     *
     * @param color color of current player
     * @param board game board state
     * @return next move
     */
    Move nextMove(Color color, Board board, long timeLimit);
  }

  private abstract static class HeuristicAlgorithm implements Algorithm {

    protected final Heuristic heuristic;

    public HeuristicAlgorithm(Heuristic heuristic) {
      this.heuristic = heuristic;
    }
  }

  /**
   * Simply choose best (depth=1)
   */
  private static class ChooseBestAlgorithm extends HeuristicAlgorithm {

    public ChooseBestAlgorithm(Heuristic heuristic) {
      super(heuristic);
    }

    public Move nextMove(Color color, Board board, long timeLimit) {
      List<Move> moves = board.getMovesFor(color);
      if (moves.size() == 0) {
        return null;
      }

      return moves.stream()
          .map(move -> evaluateMove(move, color, board))
          .max(new EvaluatedMoveComparator()).get().getMove();
    }

    private EvaluatedMove evaluateMove(Move move, Color color, Board board) {
      Board nextBoard = board.clone();
      nextBoard.doMove(move);
      return new EvaluatedMove(move, heuristic.getScore(color, nextBoard));
    }

    private static class EvaluatedMove {

      private final Move move;
      private final double score;

      public EvaluatedMove(Move move, double score) {
        this.move = move;
        this.score = score;
      }

      public Move getMove() {
        return move;
      }

      public double getScore() {
        return score;
      }
    }

    private static class EvaluatedMoveComparator implements Comparator<EvaluatedMove> {

      @Override
      public int compare(EvaluatedMove o1, EvaluatedMove o2) {
        return Double.compare(o1.getScore(), o2.getScore());
      }
    }
  }
}
