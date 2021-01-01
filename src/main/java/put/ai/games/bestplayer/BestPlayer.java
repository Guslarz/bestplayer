package put.ai.games.bestplayer;

import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeoutException;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.BinaryOperator;
import java.util.function.Predicate;

import put.ai.games.game.Board;
import put.ai.games.game.Move;
import put.ai.games.game.Player;

public class BestPlayer extends Player {

  private final Algorithm algorithm = new MiniMaxAlgorithm(new RandomHeuristic());

  @Override
  public String getName() {
    return "Zuzanna Juszczak 141238 Stanisław Kaczmarek 141240";
  }

  @Override
  public Move nextMove(Board board) {
    return algorithm.nextMove(getColor(), board, getTime());
  }

  /**
   * Object containing move and it's heuristic value
   */
  private static class EvaluatedMove {

    private double score;
    private Move move;

    public EvaluatedMove() {
      this.score = Double.NEGATIVE_INFINITY;
      this.move = null;
    }

    public EvaluatedMove(double score, Move move) {
      this.score = score;
      this.move = move;
    }

    public double getScore() {
      return score;
    }

    public Move getMove() {
      return move;
    }

    public void setScore(double score) {
      this.score = score;
    }

    public void setMove(Move move) {
      this.move = move;
    }
  }

  private static class EvaluatedMoveComparator implements Comparator<EvaluatedMove> {

    @Override
    public int compare(EvaluatedMove o1, EvaluatedMove o2) {
      return Double.compare(o1.getScore(), o2.getScore());
    }
  }

  /**
   * Exception signaling that time specified for algorithm has passed
   */
  private static class AlgorithmTimeoutException extends RuntimeException {}

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

    private final Heuristic heuristic;

    public HeuristicAlgorithm(Heuristic heuristic) {
      this.heuristic = heuristic;
    }

    @Override
    public abstract Move nextMove(Color color, Board board, long timeLimit);

    /**
     * Get heuristic score for given game state
     *
     * @param maximizer player maximizing score
     * @param current player who would make move now
     * @param board game board
     * @return heuristic score
     */
    public double getHeuristicScore(Color maximizer, Color current, Board board) {
      double multiplier = maximizer == current ? -1.0 : 1.0;
      return multiplier * heuristic.getScore(current, board);
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
      return new EvaluatedMove(getHeuristicScore(color, color, nextBoard), move);
    }
  }

  /**
   * Mini max algorithm with alpha beta pruning
   * and iterative deepening to prevent timeout
   *
   * Doesn't store tree
   */
  private static class MiniMaxAlgorithm extends HeuristicAlgorithm {

    private final static double usableTimePercent = 0.85;
    private final static long reserveTime = 50;

    public MiniMaxAlgorithm(Heuristic heuristic) {
      super(heuristic);
    }

    @Override
    public Move nextMove(Color color, Board board, long timeLimit) {
      long startTime = System.currentTimeMillis();
      long actualTimeLimit = Long.max(
          (long) (timeLimit * usableTimePercent),
          timeLimit - reserveTime
      );
      long maxTime = startTime + actualTimeLimit;
      Move bestMove = null;
      boolean run = true;

      for (int depth = 1; run; ++depth) {
        try {
          bestMove = miniMax(color, color, board,
              Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
              0, depth, maxTime).getMove();
        } catch (AlgorithmTimeoutException ex) {
          run = false;
          System.err.println(String.format(
              "Move %s found in %dms with depth %d",
              bestMove,
              System.currentTimeMillis() - startTime,
              depth - 1
          ));
        }
      }

      return bestMove;
    }

    private EvaluatedMove miniMax(Color maximizer, Color current, Board board,
                                  double alpha, double beta, int depth, int maxDepth,
                                  long maxTime) throws AlgorithmTimeoutException {
      // Throw exception if timeout
      if (System.currentTimeMillis() >= maxTime) {
        throw new AlgorithmTimeoutException();
      }

      // Evaluate board if reached maxDepth or game ended
      if (depth == maxDepth || board.getWinner(current) != null) {
        double score = getHeuristicScore(maximizer, current, board);
        return new EvaluatedMove(score, null);
      }

      EvaluatedMove result = new EvaluatedMove();
      BinaryOperator<EvaluatedMove> better;
      if (current == maximizer) {
        result.setScore(Double.NEGATIVE_INFINITY);
        better = BinaryOperator.maxBy(new EvaluatedMoveComparator());
      } else {
        result.setScore(Double.POSITIVE_INFINITY);
        better = BinaryOperator.minBy(new EvaluatedMoveComparator());
      }

      List<Move> moves = board.getMovesFor(current);
      for (Move move : moves) {
        Board nextBoard = board.clone();
        nextBoard.doMove(move);
        EvaluatedMove evaluatedMove = miniMax(maximizer, Player.getOpponent(current),
            nextBoard, alpha, beta, depth + 1, maxDepth, maxTime);
        evaluatedMove.setMove(move);
        result = better.apply(result, evaluatedMove);
        double score = result.getScore();

        if (maximizer == current) {
          if (score >= beta) {
            return result;
          }
          if (score > alpha) {
            alpha = score;
          }
        } else {
          if (score <= alpha) {
            return result;
          }
          if (score < beta) {
            beta = score;
          }
        }
      }

      return result;
    }
  }
}
