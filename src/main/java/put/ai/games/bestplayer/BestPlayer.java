package put.ai.games.bestplayer;

import put.ai.games.game.Board;
import put.ai.games.game.Move;
import put.ai.games.game.Player;

import java.util.*;

public class BestPlayer extends Player {

  private final MaxTimeCalculator maxTimeCalculator =
      new ConstantMaxTimeCalculator(100);
  private final Algorithm algorithm = new MiniMaxAlgorithm(new AltDistanceHeuristic());

  @Override
  public String getName() {
    return "Zuzanna Juszczak 141238 Stanisław Kaczmarek 141240";
  }

  @Override
  public Move nextMove(Board board) {
    long maxTime = maxTimeCalculator.getMaxTime(getTime());
    return algorithm.nextMove(getColor(), board, maxTime);
  }

  public static void main(String[] args) {
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

  /**
   * Comparator of EvaluatedMove
   */
  private static class EvaluatedMoveComparator implements Comparator<EvaluatedMove> {

    @Override
    public int compare(EvaluatedMove o1, EvaluatedMove o2) {
      return Double.compare(o1.getScore(), o2.getScore());
    }
  }

  /**
   * Interface for calculating available time for algorithm's computation
   */
  private interface MaxTimeCalculator {

    /**
     *
     * @param timeLimit time limit provided by Player class
     * @return time after which computation should be finished
     */
    long getMaxTime(long timeLimit);
  }

  /**
   * MaxTimeCalculator simply reserving some constant amount of time
   */
  private static class ConstantMaxTimeCalculator implements MaxTimeCalculator {

    private final long timeReserve;

    public ConstantMaxTimeCalculator(long timeReserve) {
      this.timeReserve = timeReserve;
    }

    @Override
    public long getMaxTime(long timeLimit) {
      return System.currentTimeMillis() + timeLimit - timeReserve;
    }
  }

  /**
   * Exception signaling that time specified for algorithm has passed
   */
  private static class AlgorithmTimeoutException extends RuntimeException {
  }

  /**
   * Interface of heuristic for game state evaluation
   */
  private interface Heuristic {

    /**
     * Evalulate board for given player
     *
     * @param maximizingPlayer maximizing player
     * @param board            game board
     * @return score as double (higher = better)
     */
    double getScore(Color maximizingPlayer, Board board);
  }

  /**
   * Random score
   */
  private static class RandomHeuristic implements Heuristic {

    private static final Random random = new Random(0xdeadbeef);

    @Override
    public double getScore(Color maximizingPlayer, Board board) {
      return random.nextDouble();
    }
  }

  /**
   * Sum of squared distances to center of points
   */
  private static class DistanceHeuristic implements Heuristic {

    @Override
    public double getScore(Color maximizingPlayer, Board board) {
      Collection<Point> playerPoints = getAllPoints(maximizingPlayer, board);
      Collection<Point> opponentPoints = getAllPoints(
          Player.getOpponent(maximizingPlayer), board);

      return sumOfSqDistance(opponentPoints) - sumOfSqDistance(playerPoints);
    }

    private static Collection<Point> getAllPoints(Color color, Board board) {
      Collection<Point> points = new ArrayList<>();
      for (int i = 0, size = board.getSize(); i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          if (board.getState(i, j) == color) {
            points.add(new Point(i, j));
          }
        }
      }

      return points;
    }

    private static double sumOfSqDistance(Collection<Point> points) {
      double x = 0;
      double y = 0;
      for (Point point : points) {
        x += point.getX();
        y += point.getY();
      }
      Point center = new Point(x / points.size(), y / points.size());

      double sum = 0;
      for (Point point : points) {
        sum += center.squaredDistanceTo(point);
      }

      return sum;
    }

    private static class Point {

      private final double x;
      private final double y;

      public Point(double x, double y) {
        this.x = x;
        this.y = y;
      }

      public double getX() {
        return x;
      }

      public double getY() {
        return y;
      }

      public double squaredDistanceTo(Point other) {
        double dx = x - other.x;
        double dy = y - other.y;
        return dx * dx + dy * dy;
      }
    }
  }

  /**
   * Counts grouped ???sheep???
   * <p>
   * Returns sum of squares of those counts divided by square of all
   * (don't want to simply loose sheep)
   */
  private static class GroupedCounterHeuristic implements Heuristic {

    @Override
    public double getScore(Color maximizingPlayer, Board board) {
      return getPlayerScore(maximizingPlayer, board) -
          getPlayerScore(Player.getOpponent(maximizingPlayer), board);
    }

    private static double getPlayerScore(Color player, Board board) {
      int size = board.getSize();
      boolean[][] visited = new boolean[size][size];

      int all = countAll(player, board, size);
      int count = 0;
      int score = 0;

      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          if (!visited[i][j]) {
            visited[i][j] = true;
            if (board.getState(i, j) == player) {
              int groupedCount = countNeighbours(player, board, i, j, visited, size);
              count += groupedCount;
              score += groupedCount * groupedCount;

              if (count == all) {
                return (double) score / (all * all);
              }
            }
          }
        }
      }

      return -1;
    }

    private static int countAll(Color player, Board board, int size) {
      int count = 0;
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          if (board.getState(i, j) == player) {
            ++count;
          }
        }
      }
      return count;
    }

    private static int countNeighbours(Color player, Board board, int x, int y,
                                       boolean[][] visited, int size) {
      int count = 1;

      for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
          if (i == 0 && j == 0) {
            continue;
          }

          int currX = x + i;
          int currY = y + j;
          if (currX < 0 || currX >= size || currY < 0 || currY >= size) {
            continue;
          }

          if (!visited[currX][currY] && board.getState(currX, currY) == player) {
            visited[x][y] = true;
            count += countNeighbours(player, board, currX, currY, visited, size);
          }
        }
      }

      return count;
    }
  }

  private static class AltDistanceHeuristic implements Heuristic {

    @Override
    public double getScore(Color maximizingPlayer, Board board) {
      Collection<Point> playerPoints = getAllPoints(maximizingPlayer, board);
      Collection<Point> opponentPoints = getAllPoints(
          Player.getOpponent(maximizingPlayer), board);

      return getSingleScore(board, playerPoints) - getSingleScore(board, opponentPoints);
    }

    private static Collection<Point> getAllPoints(Color color, Board board) {
      Collection<Point> points = new ArrayList<>();
      for (int i = 0, size = board.getSize(); i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          if (board.getState(i, j) == color) {
            points.add(new Point(i, j));
          }
        }
      }

      return points;
    }

    private static Point getCenter(Collection<Point> points) {
      int x = 0;
      int y = 0;
      for (Point point : points) {
        x += point.getX();
        y += point.getY();
      }
      return new Point(x / points.size(), y / points.size());
    }

    private int getSumOfDistances(Point center, Collection<Point> points) {
      return points.stream()
          .mapToInt(center::getDistanceTo)
          .sum();
    }

    private int getMinSumOfDistances(int pointCount) {
      int sum = 0;
      int distance = 1;
      int currCount;
      --pointCount;
      while (pointCount > 0) {
        currCount = (2 * distance - 1) * 4 + 4;
        sum += distance * currCount;
        pointCount -= currCount;
        ++distance;
      }
      return sum + pointCount * (distance - 1);
    }

    private double getQuadScore(Board board, Collection<Point> points) {
      int size = board.getSize();
      int[][] piece = new int[size + 2][size + 2];
      for (Point point : points) {
        piece[point.getX() + 1][point.getY() + 1] = 1;
      }

      int[] quadCount = new int[6];
      for (int i = 0; i <= size; ++i) {
        for (int j = 0; j <= size; ++j) {
          int count = piece[i][j] + piece[i + 1][j] + piece[i][j + 1] + piece[i + 1][j + 1];
          if (count == 2) {
            if (((piece[i][j] == 1) && (piece[i + 1][j + 1]) == 1) ||
                ((piece[i + 1][j] == 1) && (piece[i][j + 1]) == 1)) {
              ++quadCount[5];
            } else {
              ++quadCount[2];
            }
          } else {
            ++quadCount[count];
          }
        }
      }

      return (quadCount[1] - quadCount[3] - 2 * quadCount[5]) / 4.0;
    }

    private double getSingleScore(Board board, Collection<Point> points) {
      Point center = getCenter(points);
      int sumOfDistances = getSumOfDistances(center, points);
      int minSumOfDistances = getMinSumOfDistances(points.size());
      int diffOfSums = sumOfDistances - minSumOfDistances;
      double avgDistance = (double)diffOfSums / points.size();
      double quadScore = getQuadScore(board, points);
      return quadScore - (diffOfSums + avgDistance);
    }

    private static class Point {

      private final int x;
      private final int y;

      public Point(int x, int y) {
        this.x = x;
        this.y = y;
      }

      public int getX() {
        return x;
      }

      public int getY() {
        return y;
      }

      public int getDistanceTo(Point other) {
        int dx = Math.abs(x - other.x);
        int dy = Math.abs(y - other.y);
        return Integer.max(dx, dy);
      }
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
     * @param maxTime system time after which computation should be terminated
     * @return next move
     */
    Move nextMove(Color color, Board board, long maxTime);
  }

  private abstract static class HeuristicAlgorithm implements Algorithm {

    private final Heuristic heuristic;

    public HeuristicAlgorithm(Heuristic heuristic) {
      this.heuristic = heuristic;
    }

    @Override
    public abstract Move nextMove(Color player, Board board, long maxTime);

    /**
     * Get heuristic score for given game state
     *
     * @param maximizingPlayer player maximizing score
     * @param board            game board
     * @return heuristic score
     */
    public double getHeuristicScore(Color maximizingPlayer, Board board, Color winner) {
      if (winner != null) {
        if (winner == maximizingPlayer) {
          return Double.POSITIVE_INFINITY;
        } else if (winner == Color.EMPTY) {
          return 0;
        } else {
          return Double.NEGATIVE_INFINITY;
        }
      }

      return heuristic.getScore(maximizingPlayer, board);
    }
  }

  /**
   * Simply choose best (depth=1)
   */
  private static class ChooseBestAlgorithm extends HeuristicAlgorithm {

    public ChooseBestAlgorithm(Heuristic heuristic) {
      super(heuristic);
    }

    public Move nextMove(Color player, Board board, long maxTime) {
      List<Move> moves = board.getMovesFor(player);
      if (moves.size() == 0) {
        return null;
      }

      return moves.stream()
          .map(move -> evaluateMove(move, player, board))
          .max(new EvaluatedMoveComparator()).get().getMove();
    }

    private EvaluatedMove evaluateMove(Move move, Color maximizingPlayer, Board board) {
      Board nextBoard = board.clone();
      nextBoard.doMove(move);
      return new EvaluatedMove(getHeuristicScore(maximizingPlayer, nextBoard, null), move);
    }
  }

  private interface MiniMaxPredicate {

    boolean isScoreBetter(double best, double current);
    boolean shouldPrune(double best, double alpha, double beta);
    boolean shouldChangeAlpha(double best, double alpha);
    boolean shouldChangeBeta(double best, double beta);
  }

  private static class MiniMaxMinimizingPredicate implements MiniMaxPredicate {

    public boolean isScoreBetter(double best, double current) {
      return best >= current;
    }

    public boolean shouldPrune(double best, double alpha, double beta) {
      return best <= alpha;
    }

    public boolean shouldChangeAlpha(double best, double alpha) {
      return false;
    }

    public boolean shouldChangeBeta(double best, double beta) {
      return best < beta;
    }
  }

  private static class MiniMaxMaximizingPredicate implements MiniMaxPredicate {

    public boolean isScoreBetter(double best, double current) {
      return best <= current;
    }

    public boolean shouldPrune(double best, double alpha, double beta) {
      return best >= beta;
    }

    public boolean shouldChangeAlpha(double best, double alpha) {
      return best > alpha;
    }

    public boolean shouldChangeBeta(double best, double beta) {
      return false;
    }
  }

  /**
   * Mini max algorithm with alpha beta pruning
   * and iterative deepening to prevent timeout
   * <p>
   * Doesn't store tree
   */
  private static class MiniMaxAlgorithm extends HeuristicAlgorithm {

    // Stores number of nodes evaluated in single nextMove call
    private int nodeCount;

    private final static int maxAllowedDepth = 10;

    public MiniMaxAlgorithm(Heuristic heuristic) {
      super(heuristic);
    }

    @Override
    public Move nextMove(Color player, Board board, long maxTime) {
      long startTime = System.currentTimeMillis();
      Move bestMove = null;
      boolean run = true;

      nodeCount = 0;
      for (int depth = 1; run && depth <= maxAllowedDepth; ++depth) {
        try {
          bestMove = miniMax(player, player, board,
              Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY,
              0, depth, maxTime).getMove();
        } catch (AlgorithmTimeoutException ex) {
          run = false;
          System.err.println(String.format(
              "Move %s found in %dms with depth %d, %d nodes evaluated",
              bestMove, System.currentTimeMillis() - startTime,
              depth - 1, nodeCount
          ));
        }
      }

      return bestMove;
    }

    private EvaluatedMove miniMax(Color maximizingPlayer, Color currentPlayer, Board board,
                                  double alpha, double beta, int depth, int maxDepth,
                                  long maxTime) throws AlgorithmTimeoutException {
      // Throw exception if timeout
      if (System.currentTimeMillis() >= maxTime) {
        throw new AlgorithmTimeoutException();
      }

      // Evaluate board if reached maxDepth or game ended
      Color winner = board.getWinner(currentPlayer);
      if (depth == maxDepth || winner != null) {
        double score = getHeuristicScore(maximizingPlayer, board, winner);
        ++nodeCount;
        return new EvaluatedMove(score, null);
      }

      EvaluatedMove result = new EvaluatedMove();
      MiniMaxPredicate predicate;
      if (currentPlayer == maximizingPlayer) {
        result.setScore(Double.NEGATIVE_INFINITY);
        predicate = new MiniMaxMaximizingPredicate();
      } else {
        result.setScore(Double.POSITIVE_INFINITY);
        predicate = new MiniMaxMinimizingPredicate();
      }

      List<Move> moves = board.getMovesFor(currentPlayer);
      for (Move move : moves) {
        Board nextBoard = board.clone();
        nextBoard.doMove(move);
        EvaluatedMove evaluatedMove = miniMax(maximizingPlayer, Player.getOpponent(currentPlayer),
            nextBoard, alpha, beta, depth + 1, maxDepth, maxTime);
        if (predicate.isScoreBetter(result.getScore(), evaluatedMove.getScore())) {
          evaluatedMove.setMove(move);
          result = evaluatedMove;
          double score = result.getScore();
          if (predicate.shouldPrune(score, alpha, beta)) {
            return result;
          }
          if (predicate.shouldChangeAlpha(score, alpha)) {
            alpha = score;
          } else if (predicate.shouldChangeBeta(score, beta)) {
            beta = score;
          }
        }
      }

      return result;
    }
  }

  /**
   * Same as MiniMaxAlgorithm but stores tree
   */
  private static class MiniMaxTreeAlgorithm extends HeuristicAlgorithm {

    private final static int maxAllowedDepth = 10;

    private int nodeCount;

    public MiniMaxTreeAlgorithm(Heuristic heuristic) {
      super(heuristic);
    }

    @Override
    public Move nextMove(Color player, Board board, long maxTime) {
      long startTime = System.currentTimeMillis();
      Move bestMove = null;
      boolean run = true;
      nodeCount = 0;
      Tree tree = new Tree(player, board);

      for (int depth = 1; run && depth <= maxAllowedDepth; ++depth) {
        try {
          deepenTree(tree, player, maxTime);
          bestMove = miniMax(player, tree.getRoot(), Double.NEGATIVE_INFINITY,
              Double.POSITIVE_INFINITY, maxTime).getMove();
        } catch (AlgorithmTimeoutException ex) {
          run = false;
          System.err.println(String.format(
              "Move %s found in %dms with depth %d, %d nodes evaluated",
              bestMove, System.currentTimeMillis() - startTime,
              depth - 1, nodeCount
          ));
        }
      }
      return bestMove;
    }

    private void deepenTree(Tree tree, Color maximizingPlayer, long maxTime) {
      Stack<Node> nodes = new Stack<>();
      nodes.push(tree.getRoot());

      while (!nodes.empty()) {
        // Throw exception if timeout
        if (System.currentTimeMillis() >= maxTime) {
          throw new AlgorithmTimeoutException();
        }

        Node node = nodes.pop();
        if (node.getChildren().isEmpty()) {
          Board board = node.getBoard();
          if (board != null) {
            Color currentPlayer = node.getCurrentPlayer();
            Color opponent = Player.getOpponent(currentPlayer);
            for (Move move : board.getMovesFor(currentPlayer)) {
              // Throw exception if timeout
              if (System.currentTimeMillis() >= maxTime) {
                throw new AlgorithmTimeoutException();
              }

              Board nextBoard = board.clone();
              nextBoard.doMove(move);

              // Throw exception if timeout
              if (System.currentTimeMillis() >= maxTime) {
                throw new AlgorithmTimeoutException();
              }

              Color winner = nextBoard.getWinner(currentPlayer);
              Node child = new Node(move, opponent, nextBoard,
                  getHeuristicScore(maximizingPlayer, nextBoard, winner));
              node.addChild(child);
              ++nodeCount;
            }
          }
        } else {
          for (Node child : node.getChildren()) {
            nodes.push(child);
          }
        }
      }
    }

    private EvaluatedMove miniMax(Color maximizingPlayer, Node node,
                                 double alpha, double beta, long maxTime)
        throws AlgorithmTimeoutException {
      // Throw exception if timeout
      if (System.currentTimeMillis() >= maxTime) {
        throw new AlgorithmTimeoutException();
      }

      Color currentPlayer = node.getCurrentPlayer();
      EvaluatedMove result = new EvaluatedMove();
      MiniMaxPredicate predicate;
      if (currentPlayer == maximizingPlayer) {
        result.setScore(Double.NEGATIVE_INFINITY);
        predicate = new MiniMaxMaximizingPredicate();
      } else {
        result.setScore(Double.POSITIVE_INFINITY);
        predicate = new MiniMaxMinimizingPredicate();
      }

      // if no children - either not generated yet or end game node
      if (node.getChildren().isEmpty()) {
        Board board = node.getBoard();
        // Return score if game ended (no board)
        if (board == null) {
          result.setScore(node.getScore());
          return result;
        }

        // Generate all children
        for (Move move : board.getMovesFor(currentPlayer)) {
          // Throw exception if timeout
          if (System.currentTimeMillis() >= maxTime) {
            throw new AlgorithmTimeoutException();
          }

          Board nextBoard = board.clone();
          nextBoard.doMove(move);

          // Throw exception if timeout
          if (System.currentTimeMillis() >= maxTime) {
            throw new AlgorithmTimeoutException();
          }

          Color opponent = Player.getOpponent(currentPlayer);
          Color winner = board.getWinner(currentPlayer);
          Node child = new Node(move, opponent, nextBoard,
              getHeuristicScore(maximizingPlayer, nextBoard, winner));
          node.addChild(child);
          ++nodeCount;
        }

        for (Node child : node.getChildren()) {
          if (predicate.isScoreBetter(result.getScore(), child.getScore())) {
            double score = child.getScore();
            result.setMove(child.getMove());
            result.setScore(score);
            if (predicate.shouldPrune(score, alpha, beta)) {
              return result;
            }
            if (predicate.shouldChangeAlpha(score, alpha)) {
              alpha = score;
            } else if (predicate.shouldChangeBeta(score, beta)) {
              beta = score;
            }
          }
        }
      } else {
        for (Node child : node.getChildren()) {
          EvaluatedMove evaluatedMove = miniMax(maximizingPlayer, child, alpha, beta, maxTime);
          if (predicate.isScoreBetter(result.getScore(), evaluatedMove.getScore())) {
            double score = evaluatedMove.getScore();
            result.setMove(child.getMove());
            result.setScore(score);
            if (predicate.shouldPrune(score, alpha, beta)) {
              return result;
            }
            if (predicate.shouldChangeAlpha(score, alpha)) {
              alpha = score;
            } else if (predicate.shouldChangeBeta(score, beta)) {
              beta = score;
            }
          }
        }
      }

      return result;
    }

    private static class Node {

      private final Move move;
      private final Color currentPlayer;
      private double score;
      // null board signalises that game is determined as finished
      private Board board;
      private final Collection<Node> children;

      public Node(Move move, Color currentPlayer, Board board, double score) {
        this.move = move;
        this.currentPlayer = currentPlayer;
        this.board = board;
        this.score = score;
        this.children = new ArrayList<>();
      }

      public Move getMove() {
        return move;
      }

      public Color getCurrentPlayer() {
        return currentPlayer;
      }

      public double getScore() {
        return score;
      }

      public Board getBoard() {
        // Board will be evaluated only once so there's no need to store it anymore
        Board board = this.board;
        this.board = null;
        return board;
      }

      public Collection<Node> getChildren() {
        return children;
      }

      public void setScore(double score) {
        this.score = score;
      }

      public void addChild(Node child) {
        children.add(child);
      }
    }

    private static class Tree {

      private final Node root;

      public Tree(Color maximizingPlayer, Board board) {
        this.root = new Node(null, maximizingPlayer, board, 0);
      }

      public Node getRoot() {
        return root;
      }
    }
  }
}
