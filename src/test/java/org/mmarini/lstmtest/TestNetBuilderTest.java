package org.mmarini.lstmtest;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TestNetBuilderTest {

	private TestConfBuilder builder;

	@BeforeEach
	void setup() {
		builder = new TestConfBuilder(3, 3, 4);
	}

	@Test
	void testBuild() {
		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		assertNotNull(net);
//		net.init();
//		net.fit();
	}

}
