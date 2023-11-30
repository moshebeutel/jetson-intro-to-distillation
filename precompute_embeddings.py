from jetson_distillation.utils.stl10_utils import (
    precompute_clip_stl10_train_image_embeddings,
    precompute_clip_stl10_test_image_embeddings,
    precompute_clip_stl10_text_embeddings,
    get_clip_stl10_text_embeddings,
    get_stl10_train_embedding_dataset,
    get_stl10_test_embedding_dataset
)




def main():
    
    # print('precompute_clip_stl10_train_image_embeddings')
    # precompute_clip_stl10_train_image_embeddings()
    
    # print('precompute_clip_stl10_test_image_embeddings')
    # precompute_clip_stl10_test_image_embeddings()
    
    print('precompute_clip_stl10_text_embeddings')
    precompute_clip_stl10_text_embeddings()
    

    txt_emb = get_clip_stl10_text_embeddings()
    print('Got Text Embedding Tensor. Shape', txt_emb.shape)

    # train_emb_ds = get_stl10_train_embedding_dataset()
    # print('Got train embedding dataset of size', len(train_emb_ds))

    # test_emb_ds = get_stl10_test_embedding_dataset()
    # print('Got test embedding dataset of size', len(test_emb_ds))




if __name__=='__main__':
    main()