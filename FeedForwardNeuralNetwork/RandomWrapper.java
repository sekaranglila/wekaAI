package FeedForwardNeuralNetwork;

import java.io.Serializable;
import java.util.Random;


public class RandomWrapper implements Serializable
{
	private long seed;
	
	private Random rand;
	
	public RandomWrapper(long aSeed)
	{
		seed = aSeed;
		rand = new Random(aSeed);
	}
	
	public void recreate()
	{
		rand = new Random(seed);
	}

	public RandomWrapper()
	{
		this(System.currentTimeMillis());
	}	
	
	public Random getRand()
	{
		return rand;
	}

	public long getSeed()
	{
		return seed;
	}
}
