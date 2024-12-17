#ifndef USER_CODE_H
#define USER_CODE_H

#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <set>
#include "fileIterator.h"
#include "fileWriter.h"

using namespace std;

//////////////////////////////////////////////////////////////////////////////////
// MODIFY THIS SECTION
//////////////////////////////////////////////////////////////////////////////////

unordered_map<string, set<string>> static_productToHashtags;

/*
#PROMPT# 
Compare function for Sorting
*/
struct compare{
    bool operator()(const pair<int, string>& a, const pair<int, string>& b){
        if(a.first == b.first)
            return a.second < b.second;
        return a.first > b.first;
    }
};

/*
#PROMPT# 
Make a hashing function using set<string> as key for topKFrequent function
*/
struct hash_fn{
    size_t operator()(const set<string>& s) const{
        size_t hash = 0;
        for(const string& str : s){
            hash ^= hash << 1 ^ hash >> 1 ^ hash >> 3 ^ hash << 3 ^ hash >> 5 ^ hash << 5;
            for(char c : str)
                hash ^= c;
        }
        return hash;
    }
};

/*
#PROMPT# 
Map the group to customers using top k frequent words function (using min heap priority queue) and return the map
Priority queue is used to store the top k frequent hashtags for each customer
If frequencies are equal, then sort the hashtags in lexicographical order

Input is a map of customers to hashtags and the number of top tags to be returned
Output is a map of group to customers
*/
unordered_map<set<string>, vector<int>, hash_fn> topKFrequent(unordered_map<string, unordered_map<string,int>> customerToTags, int k){
    unordered_map<set<string>, vector<int>, hash_fn> groupToCustomers;
    for(auto customer : customerToTags){
        priority_queue<pair<int, string>, vector<pair<int, string>>, compare> pq;
        for(const auto& entry : customer.second){
            pq.push({entry.second, entry.first});
            if (pq.size() > k)
                pq.pop();
        }
        set<string> ans;
        while(!pq.empty()){
            ans.insert(pq.top().second);
            pq.pop();
        }
        groupToCustomers[ans].push_back(stoi(customer.first));
    }
    return groupToCustomers;
}

/**
 * @brief Modify this code to solve the assigment. Include relevant document. Mention the prompts used prefixed by #PROMPT#.
 *        
 * 
 * @param hashtags 
 * @param purchases 
 * @param prices 
 */

/*
EdgeCases I:

1. Customer to product mapping, but product not in input_product [Passed]
2. No hashtags for a product in input_product [Passed]
3. New customer with product not in input_product [Passed]
4. Multiple entries for same product id in input_product [Passed]
5. Multiple entries for same customer id and same product id [Passed]
6. Same multiple hashtags for same product id [Passed]
7. Multiple same hastags on same product entry [Passed]
8. k > number of tags or k <= 0 [Passed]
9. Include new hashtags for same product id [Passed]
*/

void groupCustomersByHashtags(fileIterator& hashtags, fileIterator& purchases,fileIterator& prices, int k, string outputFilePath){
    auto start = high_resolution_clock::now();
    unordered_map<string, unordered_map<string,int>> customerToTags;
    static_productToHashtags.clear();

    /*
    #PROMPT#
    Read the hashtags file and create a map of product to hashtags.
    */
    while(hashtags.hasNext()){
        string line = hashtags.next();
        if(line.empty()) 
            continue;

        stringstream ss(line);
        string product, hashtag;
        getline(ss, product, ',');
        if(static_productToHashtags.find(product) != static_productToHashtags.end())
            continue;

        static_productToHashtags[product];
        while(getline(ss, hashtag, ',')){
            if(hashtag.empty()) 
                continue;
            static_productToHashtags[product].insert(hashtag);
        }
    }

    /*
    #PROMPT#
    Read the purchases file and create a map of customers to hashtags. 
    For each customer, calculate the frequency of each hashtag and store it in the map.
    */
    while(purchases.hasNext()){
        string line = purchases.next();
        if(line.empty()) 
            continue;
        
        stringstream ss(line);
        string customer, product;
        getline(ss, customer, ',');
        getline(ss, product);
        if(customer.empty() || product.empty() || static_productToHashtags[product].empty()) 
            continue;
        customerToTags[customer];
        for(const string& hashtag : static_productToHashtags[product])
            customerToTags[customer][hashtag]++;
    }

    unordered_map<set<string>, vector<int>, hash_fn> groupToCustomers;
    groupToCustomers = topKFrequent(customerToTags, k);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by compute part of the function: "<< duration.count() << " microseconds" << endl;

    /*
    #PROMPT#
    Write the output to the file
    */
    for(auto group : groupToCustomers)
        writeOutputToFile(group.second, outputFilePath);
    if(groupToCustomers.empty())
        writeOutputToFile({}, outputFilePath);

    return;
}

//////////////////////////////////////////////////////////////////////////////////
// MODIFY THIS SECTION
//////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Modify this code to solve the assigment. Include relevant document. Mention the prompts used prefixed by #PROMPT#.
 *        
 * 
 * @param customerList 
 * @param purchases 
 * @param prices
 */

/*
EdgeCases II:

1. Price for product not in input_prices [Passd]
2. No purchase for customer [Passd]
3. Customer list empty [Passed]
*/
float calculateGroupAverageExpense(vector<int> customerList, fileIterator& purchases,fileIterator& prices){
    auto start = high_resolution_clock::now();

    /*
    #PROMPT# 
    Read the prices file and create a map of product to prices.
    */
    unordered_map<string, float> productToPrices;
    while(prices.hasNext()){
        string line = prices.next();
        if(line.empty()) 
            continue;      
        stringstream ss(line);
        string product, price;
        getline(ss, product, ',');
        getline(ss, price);
        productToPrices[product] = stof(price);
    }

    /*
    #PROMPT#
    Read the purchases file and create a map of customer to total expense.
    */
    unordered_map<string, float> priceMap;
    while(purchases.hasNext()){
        string line = purchases.next();
        if(line.empty()) 
            continue;
        stringstream ss(line);
        string customer, product;
        getline(ss, customer, ',');
        getline(ss, product);
        priceMap[customer] += productToPrices[product];
    }

    float totalExpense = 0;
    for(auto customer : customerList)
        totalExpense += priceMap[to_string(customer)];

    float avgExpense = totalExpense / customerList.size();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by this function: "<< duration.count() << " microseconds" << endl;

    return avgExpense;
}


//////////////////////////////////////////////////////////////////////////////////
// MODIFY THIS SECTION
//////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Modify this code to solve the assigment. Include relevant document. Mention the prompts used prefixed by #PROMPT#.
 *        
 * 
 * @param hashtags 
 * @param purchases 
 * @param prices
 * @param newHashtags
 * @param k
 * @param outputFilePath
 */

/*
EdgeCases III:

1. Customer to product mapping, but product not in input_product [Passed]
2. No hashtags for a product in input_product [Passed]
3. New customer with product not in input_product [Passed]
4. Multiple entries for same product id in input_product [Passed]
5. Multiple entries for same customer id and same product id [Passed]
6. Same multiple hashtags for same product id [Passed]
7. Multiple same hastags on same product entry [Passed]
8. k > number of tags or k <= 0 [Passed]
9. Include new hashtags for same product id [Passed] 

1. New hashtags for a product not in input_product [Passed]
2. product not in initial mapping but included dynamically [Passed]
3. Same hashtags included multiple times for a product [Passed]
*/
void groupCustomersByHashtagsForDynamicInserts(fileIterator& hashtags, fileIterator& purchases,fileIterator& prices,vector<string> newHashtags, int k, string outputFilePath){
    auto start = high_resolution_clock::now();
    unordered_map<string, set<string>> productToTags = static_productToHashtags;
    unordered_map<string, unordered_map<string,int>> customerToTags;
    
    /*
    #PROMPT#
    If productToHashtags is empty then read the hashtags file and create a map of product to hashtags.
    */
    if(productToTags.empty()){
        while (hashtags.hasNext()){
            string line = hashtags.next();
            if(line.empty()) 
                continue;
            
            stringstream ss(line);
            string product, tag;
            getline(ss, product, ',');
            productToTags[product];
            while(getline(ss, tag, ',')){
                if(tag.empty()) 
                    continue;
                productToTags[product].insert(tag);
            }
        }
    }

    /*
    #PROMPT#
    Map the products to newHashtags as well and store them in the same productToTags map.
    */
    for(auto newHashtag : newHashtags){
        stringstream ss(newHashtag);
        string product, tag;
        getline(ss, product, ',');
        if (product.empty()) 
            continue;
        while(getline(ss, tag, ',')){
            if(tag.empty()) 
                continue;
            productToTags[product].insert(tag);
        }
    }

    while(purchases.hasNext()){
        string line = purchases.next();
        if(line.empty()) 
            continue;
        stringstream ss(line);
        string customer, product;
        getline(ss, customer, ',');
        getline(ss, product);
        if(customer.empty() || product.empty() || static_productToHashtags[product].empty()) 
            continue;
        for(const string& tag : productToTags[product])
            customerToTags[customer][tag]++;
    }

    unordered_map<set<string>, vector<int>, hash_fn> finalOutput;
    finalOutput = topKFrequent(customerToTags, k);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by compute part of the function: "<< duration.count() << " microseconds" << endl;
    
    for(auto group : finalOutput)
        writeOutputToFile(group.second, outputFilePath);
    if(finalOutput.empty())
        writeOutputToFile({}, outputFilePath);
    return;
    
}



#endif // USER_CODE_H