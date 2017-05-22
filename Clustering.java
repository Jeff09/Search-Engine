import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Scanner;

public class Clustering {
 HashMap<Integer, Integer> kmean_doc2cluster = new HashMap<Integer, Integer>();
 HashMap<Integer, Integer> average_doc2cluster = new HashMap<Integer, Integer>();
 HashMap<Integer, Integer> complete_doc2cluster = new HashMap<Integer, Integer>();
 List<List<Double>> kmeans_centroids = new ArrayList<List<Double>>();
 List<List<Double>> average_centroids = new ArrayList<List<Double>>();
 List<List<Double>> complete_centriods = new ArrayList<List<Double>>(); 
 
 List<HashMap<String, Double>> complete_centroids_map = new ArrayList<HashMap<String, Double>>();
 List<HashMap<String, Double>> kmeans_centroids_map = new ArrayList<HashMap<String, Double>>();
 List<HashMap<String, Double>> average_centroids_map = new ArrayList<HashMap<String, Double>>();
 
 List<String> vac = new ArrayList<String>(); 
 
 int best_kmeans_cluster = -1;
 int best_average_cluster = -1;
 int best_complete_cluster = -1;
 
 List<Integer> best_kmeans_cluster_docs = new ArrayList<Integer>();
 List<Integer> best_complete_cluster_docs = new ArrayList<Integer>();
 List<Integer> best_average_cluster_docs = new ArrayList<Integer>();
 
 private static final String average_centroids_path = "centroids_average_sample.csv";
 private static final String complete_centroids_path = "centroids_complete_sample.csv";
 private static final String kmean_centroids_path = "centroids_kmeans_sample.csv";
 
 private static final String kmean_doc2cluster_path = "doc2cluster_kmeans_sample.csv";
 private static final String average_doc2cluster_path = "doc2cluster_average_sample.csv";
 private static final String complete_doc2cluster_path = "doc2cluster_complete_sample.csv";
 
 private static final String vac_path = "top_vac.txt";
 
 
 @SuppressWarnings("resource")
public void start() throws Exception{
	
	BufferedReader kmeans_centroids_file = new BufferedReader(new FileReader(kmean_centroids_path));
	BufferedReader average_centroids_file = new BufferedReader(new FileReader(average_centroids_path));
	BufferedReader complete_centriods_file = new BufferedReader(new FileReader(complete_centroids_path));
	
	BufferedReader kmeans_doc2cluster_file = new BufferedReader(new FileReader(kmean_doc2cluster_path));
	BufferedReader average_doc2cluster_file = new BufferedReader(new FileReader(average_doc2cluster_path));
	BufferedReader complete_doc2cluster_file = new BufferedReader(new FileReader(complete_doc2cluster_path));
	
	BufferedReader vac_file = new BufferedReader(new FileReader(vac_path));
	
	String line = null;
	while((line = kmeans_centroids_file.readLine()) != null){
		String[] e = line.split(",");
		List<Double> centroid = new ArrayList<Double>();
		for(String s : e){
			double x = Double.valueOf(s.trim());
			centroid.add(x);
		}
		kmeans_centroids.add(centroid);		
	}
	
	String line1 = null;
	while((line1=average_centroids_file.readLine()) != null){
		String[] e = line1.split(",");
		List<Double> centroid = new ArrayList<Double>();
		for(String s : e){
			double x = Double.valueOf(s.trim());
			centroid.add(x);
		}
		average_centroids.add(centroid);			
	}
	String line2 = null;
	while((line2 =complete_centriods_file.readLine()) != null){
		String[] e = line2.split(",");
		List<Double> centroid = new ArrayList<Double>();
		for(String s : e){
			double x = Double.valueOf(s.trim());
			centroid.add(x);
		}
		complete_centriods.add(centroid);			
	}
	
	String line3 = null;
	while((line3 = kmeans_doc2cluster_file.readLine()) != null ){
		String[] e = line3.split(",");		
		int doc = (int) Double.parseDouble(e[0]);
		int cluster = (int) Double.parseDouble(e[1]);
		kmean_doc2cluster.put(doc, cluster);	
	}
	
	String line4 = null;
	while((line4=average_doc2cluster_file.readLine()) != null ){
		String[] e = line4.split(",");		
		int doc = (int) Double.parseDouble(e[0]);
		int cluster = (int) Double.parseDouble(e[1]);
		average_doc2cluster.put(doc, cluster);	
	}
	
	String line5 = null;
	while((line5=complete_doc2cluster_file.readLine()) != null ){
		String[] e = line5.split(",");		
		int doc = (int) Double.parseDouble(e[0]);
		int cluster = (int) Double.parseDouble(e[1]);
		complete_doc2cluster.put(doc, cluster);	
	}
	
	String line6 = null;
	while((line6=vac_file.readLine()) != null){
		String[] e = line6.split(" ");
		for (String each:e){
			vac.add(each);
		}
	}
	
 }
 
 public  List<Integer> get_best_average_cluster_docs(){
	 for(Integer doc: average_doc2cluster.keySet()){
		 if(average_doc2cluster.get(doc) == best_average_cluster){
			 best_average_cluster_docs.add(doc);
		 }
	 }	 
	 return  best_average_cluster_docs;
 }
 
 public  List<Integer> get_best_kmeans_cluster_docs(){
	 for(Integer doc: kmean_doc2cluster.keySet()){
		 if(kmean_doc2cluster.get(doc) == best_kmeans_cluster){
			 best_kmeans_cluster_docs.add(doc);
		 }
	 }
	 return  best_kmeans_cluster_docs;
 }
 public  List<Integer> get_best_complete_cluster_docs(){	 
	 for(Integer doc: complete_doc2cluster.keySet()){
		 if(complete_doc2cluster.get(doc) == best_complete_cluster){
			 best_complete_cluster_docs.add(doc);
		 }
	 }
	 return  best_complete_cluster_docs;
 }
  
 public HashMap<String, Double> ListsToMap(List<String> vac, List<Double> centroids){
	 HashMap<String, Double> map = new LinkedHashMap<String, Double>();
	 for (int i=0; i<vac.size(); i++){
		 if (centroids.get(i)>=0){
			 map.put(vac.get(i), centroids.get(i));
		 }		 
	 }	 
	return map;	 
 }
 
 public void get_clusters_vec(){
	 
	 //List<HashMap> kmeans_centroids_map = new ArrayList<>();
	 for(List<Double> doc_kmean_centroids:kmeans_centroids){
		 HashMap<String, Double> mMap = new HashMap<String, Double>();
		 mMap = this.ListsToMap(vac, doc_kmean_centroids);
		 kmeans_centroids_map.add(mMap);
	 }
	 
	 //List<HashMap> average_centroids_map = new ArrayList<>();
	 for(List<Double> doc_average_centroids:average_centroids){
		 HashMap<String, Double> mMap = new HashMap<String, Double>();
		 mMap = this.ListsToMap(vac, doc_average_centroids);
		 average_centroids_map.add(mMap);
	 }
	 
	 //List<HashMap> complete_centroids_map = new ArrayList<>();
	 for(List<Double> doc_complete_centroids:complete_centriods){
		 HashMap<String, Double> mMap = new HashMap<String, Double>();
		 mMap = this.ListsToMap(vac, doc_complete_centroids);
		 complete_centroids_map.add(mMap);
	 } 
 } 
 
 public HashMap<String, Double> get_query_map(List<String> query){
	 HashMap<String, Double> query_map = new LinkedHashMap<String, Double>();
	 HashMap<String, Integer> tf_map = new LinkedHashMap<String, Integer>();
	 for(String term : query){
		 if(!tf_map.containsKey(term)){
			 tf_map.put(term, 1);
		 }
		 else{
			 tf_map.put(term, tf_map.get(term) + 1);
		 }
	 }
	 
	 for(String term : query){
		 query_map.put(term, (double) (1.0+Math.log10(tf_map.get(term))));
	 }
	return query_map;
 }
 
 public double get_cosine_similarity(HashMap<String, Double> que_map, HashMap<String, Double> cluster_map){
	 double dotproduct = 0;
	 double m1 = 0;
	 double m2 = 0;
	 double cossim = 0;
	 for (String term : que_map.keySet()){
		 if(cluster_map.containsKey(term)){
			dotproduct += que_map.get(term) * cluster_map.get(term);
			m1 += Math.pow(que_map.get(term), 2);
			m2 += Math.pow(cluster_map.get(term), 2);			
		 }
	 }
	 m1 = (double) Math.sqrt(m1);
	 m2 = (double) Math.sqrt(m2);
	 
	 if(m1 != 0 | m2 != 0){
		 cossim = dotproduct / (m1*m2);
	 }
	 else{
		 return 0;
	 }
	return cossim;	 
 }
 
 public void get_best_cluster(HashMap<String, Double> que_map){
	 
	 double best_kmeans_cos = 0;
	 int i = 0;
	 for(HashMap<String, Double> cluster_map:kmeans_centroids_map){
		 //System.out.println("que_map:");
		 //System.out.println(que_map);
		 //System.out.println("kmeans_cluster_map:");
		 //System.out.println(cluster_map);
		 double cos = this.get_cosine_similarity(que_map, cluster_map);
		 
		 if(cos>best_kmeans_cos){
			 best_kmeans_cos = cos;
			 best_kmeans_cluster = i;
		 }
		 i += 1;
	 }
	 
	 double best_average_cos = 0;
	 int i2 = 0;
	 for(HashMap<String, Double> cluster_map:average_centroids_map){
		 double cos = this.get_cosine_similarity(que_map, cluster_map);
		 if(cos>best_average_cos){
			 best_average_cos = cos;
			 best_average_cluster = i2;
		 }
		 i2 += 1;
	 }
	 
	 double best_complete_cos = 0;
	 int i3 = 0;
	 for(HashMap<String, Double> cluster_map:complete_centroids_map){
		 double cos = this.get_cosine_similarity(que_map, cluster_map);
		 if(cos>best_complete_cos){
			 best_complete_cos = cos;
			 best_complete_cluster = i3;
		 }
		 i3 += 1;
	 }
	 
 }
 
}

