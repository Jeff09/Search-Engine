import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class test {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
	List<String> query = Arrays.asList("swim");
		
     Clustering cc = new Clustering();
     cc.start();
     cc.get_clusters_vec();
     HashMap<String, Double> que_map = cc.get_query_map(query);
     
     cc.get_best_cluster(que_map);
     List<Integer> best_average_cluster_docs = cc.get_best_average_cluster_docs();
     List<Integer> best_kmeans_cluster_docs = cc.get_best_kmeans_cluster_docs();
     List<Integer> best_complete_cluster_docs = cc.get_best_complete_cluster_docs();
     
     System.out.println("best average cluster docs:");
     System.out.println(best_average_cluster_docs);
     
     System.out.println("best kmeans cluster docs:");
     System.out.println(best_kmeans_cluster_docs);
     
     System.out.println("best complete cluster docs:");
     System.out.println(best_complete_cluster_docs);
	}

}
